from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester


LOG = logging.getLogger("deepesense.data_fetcher")


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _extract_spacegroup(doc: Any) -> Optional[str]:
    """
    MP API docs expose symmetry/spacegroup with slightly different shapes
    depending on endpoint/version. This normalizes to a readable label.
    """
    for attr in ("symmetry", "spacegroup"):
        obj = getattr(doc, attr, None)
        if not obj:
            continue

        # dict-like
        if isinstance(obj, dict):
            symbol = obj.get("symbol") or obj.get("international_symbol")
            number = obj.get("number")
            if symbol and number:
                return f"{symbol} ({number})"
            return symbol or (str(number) if number is not None else None)

        # pydantic-like / namespace-like
        symbol = getattr(obj, "symbol", None) or getattr(obj, "international_symbol", None)
        number = getattr(obj, "number", None)
        if symbol and number is not None:
            return f"{symbol} ({number})"
        if symbol:
            return str(symbol)
        if number is not None:
            return str(number)

    return None


def fetch_mp_semiconductors(
    api_key: str,
    limit: Optional[int] = 10,
) -> pd.DataFrame:
    """
    Fetch Materials Project entries that are:
    - stable-ish: energy_above_hull < 0.05 eV/atom
    - semiconductors: 0.5 <= band_gap <= 2.5 eV

    Returns a dataframe with formula, formation energy per atom, bandgap, spacegroup.
    """
    LOG.info("Connecting to Materials Project API.")

    fields = [
        "formula_pretty",
        "formation_energy_per_atom",
        "band_gap",
        "energy_above_hull",
        "symmetry",
        "spacegroup",
        "material_id",
    ]

    criteria: Dict[str, Any] = {
        "energy_above_hull": (0.0, 0.05),
        "band_gap": (0.5, 2.5),
    }

    rows: List[Dict[str, Any]] = []

    with MPRester(api_key) as mpr:
        LOG.info(
            "Querying MP: ehull < 0.05 eV/atom, 0.5 <= bandgap <= 2.5 eV%s",
            f", limit={limit}" if limit is not None else "",
        )

        # mp-api supports a `num_chunks`/`chunk_size` iterator via `search`.
        # We rely on `num_chunks`/`chunk_size` behavior by slicing results
        # defensively in case the backend ignores the limit.
        docs = mpr.materials.summary.search(
            **criteria,
            fields=fields,
            chunk_size=200,
            num_chunks=1,
        )

        for doc in docs:
            rows.append(
                {
                    "material_id": getattr(doc, "material_id", None),
                    "formula": getattr(doc, "formula_pretty", None),
                    "formation_energy_per_atom_eV": getattr(doc, "formation_energy_per_atom", None),
                    "bandgap_eV": getattr(doc, "band_gap", None),
                    "energy_above_hull_eV": getattr(doc, "energy_above_hull", None),
                    "spacegroup": _extract_spacegroup(doc),
                }
            )
            if limit is not None and len(rows) >= limit:
                break

    df = pd.DataFrame(rows)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch stable MP semiconductors to CSV.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Dry-run limit (default: 10). Use --no-limit for full fetch.",
    )
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="Fetch without a limit (full query).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data") / "raw" / "mp_semiconductors_v1.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG).",
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)

    load_dotenv()
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        LOG.error("Missing MP_API_KEY. Create a .env file (see .env.example).")
        return 2

    limit: Optional[int] = None if args.no_limit else args.limit
    if limit is not None and limit <= 0:
        LOG.error("--limit must be a positive integer.")
        return 2

    df = fetch_mp_semiconductors(api_key=api_key, limit=limit)
    LOG.info("Fetched %d records.", len(df))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    LOG.warning("Wrote CSV to %s", out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
