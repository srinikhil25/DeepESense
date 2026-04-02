from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition


LOG = logging.getLogger("deepesense.discovery_engine")


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


def _is_ternary_or_quaternary(formula: str) -> bool:
    try:
        comp = Composition(formula)
    except Exception:
        return False
    # Pymatgen API differs across versions; use a robust element count.
    return len(comp.elements) in (3, 4)


def _extract_spacegroup(doc: Any) -> Optional[str]:
    for attr in ("symmetry", "spacegroup"):
        obj = getattr(doc, attr, None)
        if not obj:
            continue

        if isinstance(obj, dict):
            symbol = obj.get("symbol") or obj.get("international_symbol")
            number = obj.get("number")
            if symbol and number is not None:
                return f"{symbol} ({number})"
            return symbol or (str(number) if number is not None else None)

        symbol = getattr(obj, "symbol", None) or getattr(obj, "international_symbol", None)
        number = getattr(obj, "number", None)
        if symbol and number is not None:
            return f"{symbol} ({number})"
        if symbol:
            return str(symbol)
        if number is not None:
            return str(number)

    return None


def _safe_get_structure_dict(mpr: MPRester, material_id: str) -> Optional[dict]:
    """
    mp-api has changed method names over time; try a few variants.
    Returns a JSON-serializable dict (pymatgen Structure.as_dict()) or None.
    """
    # Variant A: mpr.materials.get_structure_by_material_id
    try:
        getter = getattr(getattr(mpr, "materials", None), "get_structure_by_material_id", None)
        if callable(getter):
            s = getter(material_id)
            return s.as_dict() if s is not None else None
    except Exception as e:
        LOG.debug("Structure fetch variant A failed for %s: %s", material_id, e)

    # Variant B: mpr.get_structure_by_material_id
    try:
        getter = getattr(mpr, "get_structure_by_material_id", None)
        if callable(getter):
            s = getter(material_id)
            return s.as_dict() if s is not None else None
    except Exception as e:
        LOG.debug("Structure fetch variant B failed for %s: %s", material_id, e)

    # Variant C: materials.summary.search(fields=["structure"]) sometimes works
    return None


@dataclass(frozen=True)
class DiscoveryConfig:
    ehull_max_eV: float = 0.02
    # Prefer a stable API signal that electronic properties are missing.
    # MP exposes `has_props` in Summary; when False it often indicates "electronically dark".
    require_missing_electronic_props: bool = True
    include_structures: bool = True
    limit: Optional[int] = 50  # dry-run default
    max_chunks: int = 20


def fetch_dark_stable_candidates(api_key: str, config: DiscoveryConfig) -> pd.DataFrame:
    """
    Fetch stable-ish candidates whose electronic properties are not well documented.
    Operationally, we treat "electronically dark" as missing band gap data.
    Focus on ternary/quaternary formulas (multi-element).
    """
    fields = [
        "material_id",
        "formula_pretty",
        "formation_energy_per_atom",
        "energy_above_hull",
        "band_gap",
        "symmetry",
        "structure",
        "nelements",
        "elements",
        "theoretical",
        "has_props",
    ]

    # We keep server-side filters conservative and do the "missing band gap" check client-side
    # to avoid API differences around None queries.
    criteria: Dict[str, Any] = {
        "energy_above_hull": (0.0, config.ehull_max_eV),
        # `search` prefers num_elements; `nelements` still exists as a *returned field*
        # in your Summary schema. Using num_elements here avoids the warning.
        "num_elements": (3, 4),
    }

    rows: List[Dict[str, Any]] = []
    with MPRester(api_key) as mpr:
        LOG.info(
            "Querying MP candidates: ehull<=%.3f eV/atom, num_elements in {3,4}%s",
            config.ehull_max_eV,
            f", limit={config.limit}" if config.limit is not None else "",
        )

        # Request multiple chunks in one call; `chunk_offset` is not supported in your mp-api.
        docs = mpr.materials.summary.search(
            **criteria,
            fields=fields,
            chunk_size=500,
            num_chunks=config.max_chunks,
        )

        for doc in docs:
            formula = getattr(doc, "formula_pretty", None)
            if not formula or not _is_ternary_or_quaternary(formula):
                continue

            has_props = getattr(doc, "has_props", None)
            if config.require_missing_electronic_props and has_props is True:
                continue

            mid = getattr(doc, "material_id", None)
            if not mid:
                continue

            band_gap = getattr(doc, "band_gap", None)

            structure_dict: Optional[dict] = None
            if config.include_structures:
                s = getattr(doc, "structure", None)
                if s is not None:
                    try:
                        structure_dict = s.as_dict()
                    except Exception:
                        # Some API variants may return a dict already
                        structure_dict = s if isinstance(s, dict) else None
                if structure_dict is None:
                    LOG.debug("No structure present in summary doc for %s (continuing).", mid)

            rows.append(
                {
                    "material_id": mid,
                    "formula": formula,
                    "elements": getattr(doc, "elements", None),
                    "nelements": getattr(doc, "nelements", None),
                    "formation_energy_per_atom_eV": getattr(doc, "formation_energy_per_atom", None),
                    "energy_above_hull_eV": getattr(doc, "energy_above_hull", None),
                    "bandgap_eV": band_gap,
                    "spacegroup": _extract_spacegroup(doc),
                    "is_theoretical": getattr(doc, "theoretical", None),
                    "has_props": has_props,
                    "structure": json.dumps(structure_dict) if structure_dict is not None else None,
                }
            )

            if config.limit is not None and len(rows) >= config.limit:
                break

    df = pd.DataFrame(rows)
    LOG.info("Fetched %d candidate rows (post-filtered).", len(df))
    return df


def _chemistry_bonus(elements: Sequence[str]) -> float:
    """
    Heuristic boost for combinations that are plausibly electronically interesting
    but often under-characterized: p-block metals + chalcogenides / pnictides.
    """
    elems = set(elements)
    chalc = {"S", "Se", "Te"}
    pnict = {"N", "P", "As", "Sb", "Bi"}
    pblock_metals = {"Ga", "In", "Sn", "Tl", "Pb", "Ge", "Sb", "Bi"}

    bonus = 0.0
    if elems & chalc and elems & pblock_metals:
        bonus += 1.0
    if elems & pnict and elems & pblock_metals:
        bonus += 0.6
    if ("O" in elems) and (elems & pblock_metals):
        bonus += 0.2
    return bonus


def uniqueness_score_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a `uniqueness_score` based on:
    - inverse element frequency in the candidate pool
    - bonus for chemistry motifs of interest
    """
    if df.empty:
        df["uniqueness_score"] = []
        return df

    # Normalize elements list
    def to_elems(x: Any) -> List[str]:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []
        if isinstance(x, str):
            # MP sometimes returns list-like strings; fall back to parsing comma/space separated.
            stripped = x.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    return [str(e) for e in json.loads(stripped.replace("'", '"'))]
                except Exception:
                    pass
            return [p for p in stripped.replace(",", " ").split() if p]
        if isinstance(x, (list, tuple)):
            return [str(e) for e in x]
        return [str(x)]

    elems_col = df.get("elements")
    elems_list = (
        elems_col.apply(to_elems)
        if elems_col is not None
        else df["formula"].apply(lambda f: [str(e) for e in Composition(f).elements])
    )

    element_counts = Counter(e for row in elems_list for e in row)
    total = sum(element_counts.values()) or 1

    def rarity_score(elems: Sequence[str]) -> float:
        # Sum of inverse frequencies; favors rare element combinations
        s = 0.0
        for e in elems:
            c = element_counts.get(e, 0)
            s += 0.0 if c == 0 else (total / c) ** 0.5
        return s / max(len(elems), 1)

    scores = []
    for elems in elems_list:
        scores.append(rarity_score(elems) + _chemistry_bonus(elems))

    out = df.copy()
    out["uniqueness_score"] = scores
    out = out.sort_values("uniqueness_score", ascending=False).reset_index(drop=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Unsupervised discovery: fetch and rank candidate structures.")
    parser.add_argument("--ehull-max", type=float, default=0.02, help="Max energy above hull (eV/atom).")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Dry-run limit (default: 10). Use --no-limit for full fetch.",
    )
    parser.add_argument("--no-limit", action="store_true", help="Fetch without a limit (full query).")
    parser.add_argument(
        "--allow-bandgap",
        action="store_true",
        help="Legacy flag (kept for compatibility; bandgap presence is not used as the primary darkness criterion).",
    )
    parser.add_argument(
        "--allow-has-props",
        action="store_true",
        help="Do not require missing electronic properties (include materials where has_props == True).",
    )
    parser.add_argument(
        "--no-structures",
        action="store_true",
        help="Do not attempt to fetch structures (composition-only pipeline).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data") / "raw" / "mp_candidates_v1.csv"),
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

    cfg = DiscoveryConfig(
        ehull_max_eV=float(args.ehull_max),
        require_missing_electronic_props=not args.allow_has_props,
        include_structures=not args.no_structures,
        limit=limit,
    )

    df = fetch_dark_stable_candidates(api_key=api_key, config=cfg)
    df_ranked = uniqueness_score_table(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_ranked.to_csv(out_path, index=False)
    LOG.warning("Wrote candidates to %s", out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

