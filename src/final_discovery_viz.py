from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Nature-style: muted palette, clean axes
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Professional palette (slate / sky / emerald)
COLOR_DB = "#64748b"  # slate grey — database (0 eV)
COLOR_XGB = "#38bdf8"  # sky blue — tabular
COLOR_GNN = "#059669"  # emerald green — structural


def plot_gnn_vs_xgb_correction(
    compare_csv: Path,
    out_png: Path,
    lif_annotation: str = "+1050% structural correction",
) -> None:
    df = pd.read_csv(compare_csv)
    if df.empty:
        raise ValueError(f"No rows in {compare_csv}")

    required = {"formula", "refit_xgb_tabular_eV", "gnn_predicted_eV"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["formula_display"] = df["formula"].astype(str)
    df["database_eV"] = 0.0

    n = len(df)
    x = np.arange(n, dtype=float)
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    ax.bar(x - width, df["database_eV"], width, label="Database (MP)", color=COLOR_DB, edgecolor="white", linewidth=0.5)
    ax.bar(x, df["refit_xgb_tabular_eV"], width, label="XGBoost (tabular)", color=COLOR_XGB, edgecolor="white", linewidth=0.5)
    ax.bar(x + width, df["gnn_predicted_eV"], width, label="GNN (structural)", color=COLOR_GNN, edgecolor="white", linewidth=0.5)

    ax.axhline(0.5, color="#1e293b", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)
    ax.text(
        n - 0.02,
        0.52,
        "Semiconductor threshold (0.5 eV)",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#334155",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["formula_display"], rotation=35, ha="right")
    ax.set_ylabel("Bandgap (eV)")
    ax.set_xlabel("Material")
    ax.set_title("Database vs tabular vs structural bandgap estimates (key discoveries)")
    ax.legend(loc="upper left", frameon=False)

    ymax = float(
        max(
            df["gnn_predicted_eV"].max(),
            df["refit_xgb_tabular_eV"].max(),
            0.6,
        )
        * 1.08
    )
    ax.set_ylim(0.0, ymax)

    # LiF annotation (GNN vs tabular structural correction)
    lif_idx = np.flatnonzero(df["formula_display"].str.upper().to_numpy() == "LIF")
    if lif_idx.size:
        i = int(lif_idx[0])
        xi = x[i] + width
        y_gnn = float(df["gnn_predicted_eV"].iloc[i])
        ax.annotate(
            lif_annotation,
            xy=(xi, y_gnn),
            xytext=(min(xi + 1.2, n - 0.4), y_gnn * 0.55),
            fontsize=8,
            color="#0f766e",
            arrowprops=dict(arrowstyle="-|>", color="#0f766e", lw=0.8, shrinkA=0, shrinkB=4),
        )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grouped bar chart: database vs XGBoost vs GNN bandgaps for discovery materials."
    )
    parser.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=str(Path("results") / "compare_gnn_xgboost_top10_v1.csv"),
        help="Comparison CSV from compare_gnn_xgboost.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("results") / "figures" / "gnn_vs_xgb_correction.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--lif-note",
        type=str,
        default="+1050% structural correction",
        help="Annotation text for LiF.",
    )
    args = parser.parse_args()

    plot_gnn_vs_xgb_correction(
        compare_csv=Path(args.inp),
        out_png=Path(args.out),
        lif_annotation=args.lif_note,
    )
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
