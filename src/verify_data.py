from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element


LOG = logging.getLogger("deepesense.verify_data")


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


@dataclass(frozen=True)
class VerifyConfig:
    min_volume_per_atom: float = 0.0  # Å^3/atom; reject <= 0 as unphysical


def _parse_structure(structure_json: Any) -> Optional[Structure]:
    if structure_json is None or (isinstance(structure_json, float) and pd.isna(structure_json)):
        return None
    if isinstance(structure_json, dict):
        try:
            return Structure.from_dict(structure_json)
        except Exception:
            return None
    if isinstance(structure_json, str):
        s = structure_json.strip()
        if not s:
            return None
        try:
            return Structure.from_dict(json.loads(s))
        except Exception:
            return None
    return None


def _compute_atomic_volume(struct: Structure) -> float:
    # Å^3 / atom
    return float(struct.lattice.volume) / float(len(struct))


def _utility_tier(bandgap_eV: Any) -> str:
    if bandgap_eV is None or (isinstance(bandgap_eV, float) and pd.isna(bandgap_eV)):
        return "Unknown (Eg missing)"
    try:
        eg = float(bandgap_eV)
    except Exception:
        return "Unknown (Eg unparsable)"

    if eg <= 0.0:
        return "Tier 0: Electronic dark matter (Eg=0)"
    if eg <= 0.5:
        return "Tier 1: Narrow-gap (0<Eg≤0.5)"
    if eg <= 2.5:
        return "Tier 2: Semiconductor (0.5<Eg≤2.5)"
    return "Tier 3: Wide-gap (Eg>2.5)"


def validate_and_report(
    inp_csv: Path,
    out_clean_csv: Path,
    report_md: Path,
    figures_dir: Path,
    cfg: VerifyConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(inp_csv)
    n0 = len(df)

    # ---- integrity: required columns
    required_cols = ["material_id", "formula", "structure"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # ---- integrity: null checks
    null_formula = df["formula"].isna() | (df["formula"].astype(str).str.strip() == "")
    null_structure = df["structure"].isna() | (df["structure"].astype(str).str.strip() == "")
    df = df.loc[~null_formula & ~null_structure].copy()

    # ---- parse structures and physics sanity checks
    LOG.info("Parsing structures and computing atomic volume.")
    df["_structure_obj"] = df["structure"].apply(_parse_structure)
    parsed_ok = df["_structure_obj"].notna()
    df = df.loc[parsed_ok].copy()

    df["atomic_volume_A3_per_atom"] = df["_structure_obj"].apply(_compute_atomic_volume)
    df = df.loc[df["atomic_volume_A3_per_atom"] > cfg.min_volume_per_atom].copy()

    # (Optional) keep lattice volume too for debugging
    df["cell_volume_A3"] = df["_structure_obj"].apply(lambda s: float(s.lattice.volume))
    df["nsites"] = df["_structure_obj"].apply(lambda s: int(len(s)))

    # ---- tiers
    if "bandgap_eV" in df.columns:
        df["utility_tier"] = df["bandgap_eV"].apply(_utility_tier)
    else:
        df["utility_tier"] = "Unknown (no bandgap_eV column)"

    # ---- Chemical diversity (element counts)
    element_counts: Dict[str, int] = {}
    for s in df["_structure_obj"]:
        for el in s.composition.elements:
            sym = str(el)
            element_counts[sym] = element_counts.get(sym, 0) + 1

    # Exotic = rare in dataset; tie-break by higher Z
    exotic_sorted = sorted(
        element_counts.items(),
        key=lambda kv: (kv[1], -Element(kv[0]).Z),
    )
    top10_exotic = exotic_sorted[:10]

    # ---- Figures
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Bar: materials per tier
    tier_counts = df["utility_tier"].value_counts().sort_index()
    fig1 = plt.figure(figsize=(10, 4.8))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.bar(tier_counts.index, tier_counts.values)
    ax1.set_title("Materials per Utility Tier")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=25)
    fig1.tight_layout()
    fig1_path = figures_dir / "materials_per_utility_tier.png"
    fig1.savefig(fig1_path, dpi=180)
    plt.close(fig1)

    # Scatter: atomic volume vs formation energy
    fig2 = plt.figure(figsize=(6.6, 5.2))
    ax2 = fig2.add_subplot(1, 1, 1)
    if "formation_energy_per_atom_eV" in df.columns:
        ax2.scatter(
            df["atomic_volume_A3_per_atom"],
            df["formation_energy_per_atom_eV"],
            s=18,
            alpha=0.75,
        )
        ax2.set_ylabel("Formation energy per atom (eV)")
    else:
        ax2.scatter(df["atomic_volume_A3_per_atom"], [0] * len(df), s=18, alpha=0.75)
        ax2.set_ylabel("(missing formation_energy_per_atom_eV)")
    ax2.set_xlabel("Atomic volume (Å³/atom)")
    ax2.set_title("Atomic Volume vs Formation Energy")
    fig2.tight_layout()
    fig2_path = figures_dir / "atomic_volume_vs_formation_energy.png"
    fig2.savefig(fig2_path, dpi=180)
    plt.close(fig2)

    # Histogram: bandgaps
    fig3 = plt.figure(figsize=(6.6, 5.2))
    ax3 = fig3.add_subplot(1, 1, 1)
    if "bandgap_eV" in df.columns:
        bg = pd.to_numeric(df["bandgap_eV"], errors="coerce").dropna()
        ax3.hist(bg, bins=40)
        ax3.set_xlabel("Band gap (eV)")
        ax3.set_ylabel("Count")
        ax3.set_title("Bandgap Distribution (Electronic Dark Matter peak at 0 eV)")
    else:
        ax3.text(0.5, 0.5, "bandgap_eV column missing", ha="center", va="center")
        ax3.set_axis_off()
    fig3.tight_layout()
    fig3_path = figures_dir / "bandgap_histogram.png"
    fig3.savefig(fig3_path, dpi=180)
    plt.close(fig3)

    # ---- Write cleaned CSV (drop non-serializable structure objects)
    out_clean_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.drop(columns=["_structure_obj"])
    df_out.to_csv(out_clean_csv, index=False)

    # ---- Report (Markdown)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    removed = n0 - len(df)
    null_formula_n = int(null_formula.sum())
    null_structure_n = int(null_structure.sum())

    summary = {
        "input_rows": int(n0),
        "output_rows": int(len(df)),
        "removed_rows_total": int(removed),
        "null_formula_rows_in_input": null_formula_n,
        "null_structure_rows_in_input": null_structure_n,
        "max_ehull_eV": float(pd.to_numeric(df.get("energy_above_hull_eV", pd.Series(dtype=float)), errors="coerce").max())
        if "energy_above_hull_eV" in df.columns
        else None,
        "bandgap_zero_count": int((pd.to_numeric(df.get("bandgap_eV", pd.Series(dtype=float)), errors="coerce") == 0.0).sum())
        if "bandgap_eV" in df.columns
        else None,
        "top10_exotic_elements": top10_exotic,
    }

    md = []
    md.append("# DeepESense Validation Report\n")
    md.append(f"- **Input**: `{inp_csv.as_posix()}`\n")
    md.append(f"- **Cleaned output**: `{out_clean_csv.as_posix()}`\n")
    md.append(f"- **Rows**: {summary['input_rows']} → {summary['output_rows']} (removed {summary['removed_rows_total']})\n")
    md.append(f"- **Null formulas in input**: {summary['null_formula_rows_in_input']}\n")
    md.append(f"- **Null structures in input**: {summary['null_structure_rows_in_input']}\n")
    if summary["bandgap_zero_count"] is not None:
        md.append(f"- **Bandgap = 0.0 eV count**: {summary['bandgap_zero_count']}\n")
    md.append("\n## Materials per Utility Tier\n")
    md.append(f"![Materials per Utility Tier]({(figures_dir / fig1_path.name).as_posix()})\n")
    md.append("\n## Atomic Volume vs Formation Energy\n")
    md.append(f"![Atomic Volume vs Formation Energy]({(figures_dir / fig2_path.name).as_posix()})\n")
    md.append("\n## Bandgap Histogram\n")
    md.append(f"![Bandgap Histogram]({(figures_dir / fig3_path.name).as_posix()})\n")
    md.append("\n## Chemical Diversity: Top 10 Exotic Elements (rarest in dataset)\n")
    md.append("| Element | Count |\n|---:|---:|\n")
    for el, c in top10_exotic:
        md.append(f"| {el} | {c} |\n")

    report_md.write_text("".join(md), encoding="utf-8")
    LOG.warning("Wrote validation report to %s", report_md.as_posix())

    return df_out, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DeepESense candidate dataset and generate report.")
    parser.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=str(Path("data") / "raw" / "mp_candidates_v1.csv"),
        help="Input candidates CSV.",
    )
    parser.add_argument(
        "--out-clean",
        type=str,
        default=str(Path("data") / "processed" / "mp_candidates_v1_clean.csv"),
        help="Output cleaned CSV.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(Path("reports") / "deepesense_validation_report_v1.md"),
        help="Output validation report (Markdown).",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(Path("reports") / "figures" / "validation_v1"),
        help="Directory to write figures.",
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

    validate_and_report(
        inp_csv=Path(args.inp),
        out_clean_csv=Path(args.out_clean),
        report_md=Path(args.report),
        figures_dir=Path(args.figures_dir),
        cfg=VerifyConfig(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

