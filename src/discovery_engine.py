from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition


LOG = logging.getLogger("deepesense.discovery_engine")
REFRACTORY = {"W", "Ta", "Mo", "Nb", "Re"}


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _extract_spacegroup(doc: Any) -> Optional[str]:
    obj = getattr(doc, "symmetry", None)
    if not obj:
        return None
    if isinstance(obj, dict):
        symbol = obj.get("symbol") or obj.get("international_symbol")
        number = obj.get("number")
    else:
        symbol = getattr(obj, "symbol", None) or getattr(obj, "international_symbol", None)
        number = getattr(obj, "number", None)
    if symbol and number is not None:
        return f"{symbol} ({number})"
    return symbol or (str(number) if number is not None else None)


def _extract_elements(formula: Optional[str]) -> List[str]:
    if not formula:
        return []
    try:
        return [str(e) for e in Composition(formula).elements]
    except Exception:
        return []


def utility_tier_from_formula(formula: Optional[str]) -> str:
    elems = _extract_elements(formula)
    if not elems:
        return "Tier 1 (Standard)"

    z_values = [Composition(el).elements[0].Z for el in elems]
    if any(z > 83 for z in z_values):
        return "Tier 4 (Extreme)"
    if any(el in REFRACTORY for el in elems):
        return "Tier 3 (Refractory)"
    if any(57 <= z <= 71 for z in z_values):
        return "Tier 2 (Exotic)"
    return "Tier 1 (Standard)"


def _write_progress(rows: List[Dict[str, Any]], out_csv: Path, append: bool) -> None:
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = not append or (append and not out_csv.exists())
    pd.DataFrame(rows).to_csv(out_csv, index=False, mode="a" if append else "w", header=header)


def _write_summary(
    summary_path: Path,
    partial: bool,
    error_msg: Optional[str],
    total: int,
    missing_bg: int,
    tier_counts: Dict[str, int],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    missing_bg_pct = (100.0 * missing_bg / total) if total else 0.0

    lines = [
        "DeepESense Data Summary\n",
        f"status: {'PARTIAL' if partial else 'COMPLETE'}\n",
        f"total_materials: {total}\n",
        f"missing_band_gap_count: {int(missing_bg)}\n",
        f"missing_band_gap_percent: {missing_bg_pct:.2f}\n",
        "\nMaterials by Utility Tier:\n",
    ]
    for tier, cnt in sorted(tier_counts.items()):
        lines.append(f"- {tier}: {int(cnt)}\n")
    if error_msg:
        lines.append(f"\nlast_error: {error_msg}\n")
    summary_path.write_text("".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class AcquisitionConfig:
    ehull_max_eV: float = 0.05
    batch_size: int = 1000
    save_every: int = 1000
    max_records: Optional[int] = None


def fetch_master_dataset(api_key: str, cfg: AcquisitionConfig, out_csv: Path, summary_path: Path) -> int:
    fields = [
        "material_id",
        "formula_pretty",
        "structure",
        "band_gap",
        "formation_energy_per_atom",
        "theoretical",
        "symmetry",
        "energy_above_hull",
    ]
    criteria: Dict[str, Any] = {"energy_above_hull": (0.0, cfg.ehull_max_eV)}
    rows_buffer: List[Dict[str, Any]] = []
    total_saved = 0
    missing_bg_count = 0
    tier_counts: Dict[str, int] = {}
    partial = False
    last_error: Optional[str] = None
    is_first_flush = True

    # Start a fresh file for this run, then append incrementally.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    try:
        with MPRester(api_key) as mpr:
            LOG.info("Starting full acquisition: ehull < %.3f eV/atom", cfg.ehull_max_eV)
            docs = mpr.materials.summary.search(
                **criteria,
                fields=fields,
                chunk_size=cfg.batch_size,
                num_chunks=None,
            )

            for idx, doc in enumerate(docs, start=1):
                formula = getattr(doc, "formula_pretty", None)
                mid = getattr(doc, "material_id", None)
                if not mid or not formula:
                    continue

                s = getattr(doc, "structure", None)
                if s is None:
                    continue
                try:
                    structure_dict = s.as_dict()
                except Exception:
                    structure_dict = s if isinstance(s, dict) else None
                if structure_dict is None:
                    continue

                row = {
                    "material_id": mid,
                    "formula": formula,
                    "structure": json.dumps(structure_dict),
                    "band_gap": getattr(doc, "band_gap", None),
                    "formation_energy_per_atom": getattr(doc, "formation_energy_per_atom", None),
                    "is_theoretical": getattr(doc, "theoretical", None),
                    "symmetry": _extract_spacegroup(doc),
                    "energy_above_hull": getattr(doc, "energy_above_hull", None),
                }
                row["utility_tier"] = utility_tier_from_formula(row["formula"])
                rows_buffer.append(row)
                total_saved += 1
                if row["band_gap"] is None:
                    missing_bg_count += 1
                tier = row["utility_tier"]
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

                if len(rows_buffer) >= cfg.save_every:
                    _write_progress(rows_buffer, out_csv, append=not is_first_flush)
                    is_first_flush = False
                    rows_buffer.clear()
                    LOG.info("Checkpoint saved: %d records", total_saved)

                if cfg.max_records is not None and total_saved >= cfg.max_records:
                    LOG.warning("Reached max-records=%d; stopping early.", cfg.max_records)
                    break

    except Exception as exc:
        partial = True
        last_error = str(exc)
        LOG.exception("Acquisition interrupted; writing partial progress.")
    finally:
        _write_progress(rows_buffer, out_csv, append=not is_first_flush)
        _write_summary(
            summary_path=summary_path,
            partial=partial,
            error_msg=last_error,
            total=total_saved,
            missing_bg=missing_bg_count,
            tier_counts=tier_counts,
        )

    return total_saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Full Data Acquisition for DeepESense.")
    parser.add_argument("--ehull-max", type=float, default=0.05, help="Max energy above hull (eV/atom).")
    parser.add_argument("--batch-size", type=int, default=1000, help="API batch/chunk size.")
    parser.add_argument("--save-every", type=int, default=1000, help="Checkpoint CSV write frequency.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data") / "raw" / "deepesense_master_v1.csv"),
        help="Output master CSV path.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=str(Path("logs") / "data_summary.txt"),
        help="Output summary report path.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for test runs. Omit for full acquisition.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.batch_size <= 0 or args.save_every <= 0:
        LOG.error("--batch-size and --save-every must be positive integers.")
        return 2

    load_dotenv()
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        LOG.error("Missing MP_API_KEY. Create a .env file (see .env.example).")
        return 2

    cfg = AcquisitionConfig(
        ehull_max_eV=float(args.ehull_max),
        batch_size=int(args.batch_size),
        save_every=int(args.save_every),
        max_records=args.max_records,
    )
    total_saved = fetch_master_dataset(
        api_key=api_key,
        cfg=cfg,
        out_csv=Path(args.out),
        summary_path=Path(args.summary),
    )
    LOG.warning("Saved %d materials to %s", total_saved, Path(args.out).as_posix())
    LOG.warning("Summary written to %s", Path(args.summary).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

