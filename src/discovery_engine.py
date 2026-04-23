from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# Default per-pass element-count ranges. Each pass issues one MP search with
# ``num_elements`` constrained to the range, so no single response ever tries
# to buffer the full 77k-doc catalog at once. The ranges below were picked so
# each bucket lands in the 3k–18k doc range for the default ehull cut of 0.05
# eV/atom — comfortable for both RAM and orjson's parse buffer.
DEFAULT_NELEMENTS_PASSES: Tuple[Tuple[int, int], ...] = (
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 12),  # heavy-alloy tail; usually <1k docs
)


@dataclass(frozen=True)
class AcquisitionConfig:
    ehull_max_eV: float = 0.05
    batch_size: int = 1000
    save_every: int = 1000
    max_records: Optional[int] = None
    nelements_passes: Tuple[Tuple[int, int], ...] = DEFAULT_NELEMENTS_PASSES


def _run_search_pass(
    mpr: "MPRester",
    criteria: Dict[str, Any],
    fields: List[str],
    cfg: AcquisitionConfig,
    out_csv: Path,
    state: Dict[str, Any],
) -> None:
    """
    Run one ``materials.summary.search`` call and stream its rows into the
    on-disk CSV via the shared ``state`` dict. Keeping state external lets the
    outer per-nelements loop resume cleanly after every pass and guarantees
    that a crash mid-pass still leaves every prior pass fully persisted.
    """
    docs = mpr.materials.summary.search(
        **criteria,
        fields=fields,
        chunk_size=cfg.batch_size,
        num_chunks=None,
    )
    try:
        for doc in docs:
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

            state["rows_buffer"].append(row)
            state["total_saved"] += 1
            if row["band_gap"] is None:
                state["missing_bg_count"] += 1
            tier = row["utility_tier"]
            state["tier_counts"][tier] = state["tier_counts"].get(tier, 0) + 1

            if len(state["rows_buffer"]) >= cfg.save_every:
                _write_progress(state["rows_buffer"], out_csv, append=not state["is_first_flush"])
                state["is_first_flush"] = False
                state["rows_buffer"].clear()
                LOG.info("Checkpoint saved: %d records", state["total_saved"])

            if cfg.max_records is not None and state["total_saved"] >= cfg.max_records:
                LOG.warning("Reached max-records=%d; stopping early.", cfg.max_records)
                state["stop_early"] = True
                return
    finally:
        # Flush any tail rows from this pass before the docs list is released,
        # so a mid-next-pass crash never loses this pass's data.
        if state["rows_buffer"]:
            _write_progress(state["rows_buffer"], out_csv, append=not state["is_first_flush"])
            state["is_first_flush"] = False
            state["rows_buffer"].clear()
        # Explicitly drop the docs list and force GC so the next pass starts
        # with a clean allocator — critical on Windows under pagefile pressure.
        del docs
        gc.collect()


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
    base_criteria: Dict[str, Any] = {"energy_above_hull": (0.0, cfg.ehull_max_eV)}

    state: Dict[str, Any] = {
        "rows_buffer": [],
        "total_saved": 0,
        "missing_bg_count": 0,
        "tier_counts": {},
        "is_first_flush": True,
        "stop_early": False,
    }
    partial = False
    last_error: Optional[str] = None

    # Start a fresh file for this run, then append incrementally.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    try:
        with MPRester(api_key) as mpr:
            LOG.warning(
                "Starting chunked acquisition: ehull < %.3f eV/atom across %d nelements passes",
                cfg.ehull_max_eV, len(cfg.nelements_passes),
            )
            for pass_idx, (ne_lo, ne_hi) in enumerate(cfg.nelements_passes, start=1):
                if state["stop_early"]:
                    break
                pass_criteria = dict(base_criteria)
                pass_criteria["num_elements"] = (int(ne_lo), int(ne_hi))
                LOG.warning(
                    "Pass %d/%d: num_elements=%s (rows so far: %d)",
                    pass_idx, len(cfg.nelements_passes),
                    pass_criteria["num_elements"], state["total_saved"],
                )
                _run_search_pass(mpr, pass_criteria, fields, cfg, out_csv, state)
                LOG.warning(
                    "Pass %d/%d done. Total rows saved: %d",
                    pass_idx, len(cfg.nelements_passes), state["total_saved"],
                )

    except Exception as exc:
        partial = True
        last_error = str(exc)
        LOG.exception("Acquisition interrupted; writing partial progress.")
    finally:
        if state["rows_buffer"]:
            _write_progress(state["rows_buffer"], out_csv, append=not state["is_first_flush"])
            state["rows_buffer"].clear()
        _write_summary(
            summary_path=summary_path,
            partial=partial,
            error_msg=last_error,
            total=state["total_saved"],
            missing_bg=state["missing_bg_count"],
            tier_counts=state["tier_counts"],
        )

    return state["total_saved"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Full Data Acquisition for DeepESense.")
    parser.add_argument("--ehull-max", type=float, default=0.05, help="Max energy above hull (eV/atom).")
    parser.add_argument("--batch-size", type=int, default=1000, help="API batch/chunk size.")
    parser.add_argument("--save-every", type=int, default=1000, help="Checkpoint CSV write frequency.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data") / "raw" / "deepesense_candidates.csv"),
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

