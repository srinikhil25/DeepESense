from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler


LOG = logging.getLogger("deepesense.feature_analysis")


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    drop_cols = {"bandgap_eV", "band_gap"}
    return [c for c in numeric_cols if c not in drop_cols]


def _select_bandgap_column(df: pd.DataFrame) -> str:
    if "bandgap_eV" in df.columns:
        return "bandgap_eV"
    if "band_gap" in df.columns:
        return "band_gap"
    raise ValueError("No bandgap column found. Expected 'bandgap_eV' or 'band_gap'.")


def run_feature_analysis(inp_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(inp_csv)
    if df.empty:
        raise ValueError(f"Input is empty: {inp_csv}")

    bg_col = _select_bandgap_column(df)
    df[bg_col] = pd.to_numeric(df[bg_col], errors="coerce")
    df = df.dropna(subset=[bg_col]).copy()

    labeled = df[df[bg_col] > 0].copy()
    dark = df[df[bg_col] == 0].copy()
    if labeled.empty:
        raise ValueError("No labeled rows found (bandgap > 0).")

    feature_cols = _numeric_feature_columns(labeled)
    if not feature_cols:
        raise ValueError("No numeric feature columns available for analysis.")

    # Pearson correlation
    pearson_corr = labeled[feature_cols].corrwith(labeled[bg_col], method="pearson").dropna()
    top20_corr = pearson_corr.reindex(pearson_corr.abs().sort_values(ascending=False).head(20).index)

    # Mutual information (non-linear)
    x = labeled[feature_cols].fillna(labeled[feature_cols].median(numeric_only=True))
    y = labeled[bg_col]
    mi_scores = mutual_info_regression(x, y, random_state=42)
    mi = pd.Series(mi_scores, index=feature_cols, name="mutual_information")
    mi_scaled = pd.Series(
        MinMaxScaler().fit_transform(mi.values.reshape(-1, 1)).ravel(),
        index=feature_cols,
        name="mi_scaled_0_1",
    )

    # Combine rankings for "physical drivers"
    ranking = pd.DataFrame(
        {
            "pearson": pearson_corr,
            "pearson_abs": pearson_corr.abs(),
            "mutual_information": mi,
            "mi_scaled_0_1": mi_scaled,
        }
    ).fillna(0.0)
    ranking["combined_score"] = 0.5 * ranking["pearson_abs"] + 0.5 * ranking["mi_scaled_0_1"]
    top5_drivers = ranking.sort_values("combined_score", ascending=False).head(5)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save tables
    pearson_corr.sort_values(key=lambda s: s.abs(), ascending=False).to_csv(out_dir / "pearson_correlations.csv")
    ranking.sort_values("combined_score", ascending=False).to_csv(out_dir / "feature_driver_ranking.csv")

    # Heatmap of top 20 correlations
    heatmap_df = top20_corr.to_frame(name="pearson_r_with_bandgap")
    fig, ax = plt.subplots(figsize=(6.5, 8.0))
    im = ax.imshow(heatmap_df.values, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)
    ax.set_xticks([0])
    ax.set_xticklabels(["Pearson r"])
    ax.set_title("Top 20 Feature Correlations with Bandgap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation coefficient (r)")
    fig.tight_layout()
    fig.savefig(out_dir / "top20_feature_correlation_heatmap.png", dpi=180)
    plt.close(fig)

    # Summary report
    summary_path = out_dir / "feature_analysis_summary.txt"
    lines = [
        "DeepESense Feature Analysis Summary\n",
        f"input_csv: {inp_csv.as_posix()}\n",
        f"total_rows: {len(df)}\n",
        f"labeled_rows (bandgap > 0): {len(labeled)}\n",
        f"dark_rows (bandgap == 0): {len(dark)}\n",
        f"num_numeric_features_analyzed: {len(feature_cols)}\n",
        "\nTop 5 Physical Drivers (combined Pearson + Mutual Information):\n",
    ]
    for feat, row in top5_drivers.iterrows():
        lines.append(
            f"- {feat}: pearson={row['pearson']:.4f}, "
            f"|r|={row['pearson_abs']:.4f}, MI={row['mutual_information']:.4f}, "
            f"MI_scaled={row['mi_scaled_0_1']:.4f}, combined={row['combined_score']:.4f}\n"
        )
    summary_path.write_text("".join(lines), encoding="utf-8")

    LOG.warning("Wrote analysis outputs to %s", out_dir.as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze feature-bandgap relationships for DeepESense.")
    parser.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=str(Path("data") / "processed" / "deepesense_features_v1.csv"),
        help="Input features CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("reports") / "feature_analysis_v1"),
        help="Directory for analysis outputs.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    run_feature_analysis(inp_csv=Path(args.inp), out_dir=Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

