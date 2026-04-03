from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOG = logging.getLogger("deepesense.visualize_discoveries")


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _select_bandgap_column(df: pd.DataFrame) -> str:
    if "bandgap_eV" in df.columns:
        return "bandgap_eV"
    if "band_gap" in df.columns:
        return "band_gap"
    raise ValueError("No bandgap column found in master CSV.")


def visualize_discoveries(
    master_csv: Path,
    discoveries_csv: Path,
    figures_dir: Path,
    summary_md: Path,
) -> None:
    master = pd.read_csv(master_csv)
    discoveries = pd.read_csv(discoveries_csv)
    if master.empty or discoveries.empty:
        raise ValueError("Master or discoveries CSV is empty.")

    bg_col = _select_bandgap_column(master)
    master[bg_col] = pd.to_numeric(master[bg_col], errors="coerce")

    # Join master (old bandgaps, tiers) with discoveries (predicted bandgaps, scores)
    merged = discoveries.merge(master, on="material_id", suffixes=("", "_master"), how="left")

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Migration Plot (old vs new bandgaps)
    old_bg = pd.to_numeric(merged[bg_col], errors="coerce").fillna(0.0)
    new_bg = pd.to_numeric(merged["predicted_latent_bandgap_eV"], errors="coerce").fillna(0.0)

    fig1, ax1 = plt.subplots(figsize=(7.0, 4.5))
    bins = 30
    ax1.hist(old_bg, bins=bins, alpha=0.6, label="Old bandgaps (original)", color="C0")
    ax1.hist(new_bg, bins=bins, alpha=0.6, label="New bandgaps (DeepESense)", color="C1")
    ax1.set_xlabel("Bandgap (eV)")
    ax1.set_ylabel("Count")
    ax1.set_title("Migration Plot: Old vs New Bandgaps for Discovered Materials")
    ax1.legend()
    fig1.tight_layout()
    mig_path = figures_dir / "migration_plot_bandgaps.png"
    fig1.savefig(mig_path, dpi=180)
    plt.close(fig1)

    # Plot 2: Discovery Landscape
    tiers = merged.get("utility_tier", discoveries.get("utility_tier"))
    merged["utility_tier"] = tiers

    fig2, ax2 = plt.subplots(figsize=(7.0, 5.5))
    scatter = ax2.scatter(
        merged["predicted_latent_bandgap_eV"],
        merged["discovery_score"],
        c=pd.factorize(merged["utility_tier"])[0],
        cmap="tab10",
        s=35,
        alpha=0.8,
    )
    ax2.set_xlabel("Predicted Bandgap (eV)")
    ax2.set_ylabel("Discovery Score")
    ax2.set_title("Discovery Landscape: Predicted Bandgap vs Discovery Score")
    # Legend from unique tiers
    handles, _ = scatter.legend_elements()
    unique_tiers = list(pd.Series(merged["utility_tier"]).astype(str).unique())
    ax2.legend(handles, unique_tiers, title="Utility Tier", fontsize=8)
    fig2.tight_layout()
    land_path = figures_dir / "discovery_landscape.png"
    fig2.savefig(land_path, dpi=180)
    plt.close(fig2)

    # Table: Top 5 discoveries per tier
    merged_sorted = merged.sort_values("discovery_score", ascending=False)
    grouped = merged_sorted.groupby("utility_tier", dropna=False)

    summary_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DeepESense Discovery Summary\n",
        f"- **Master CSV**: `{master_csv.as_posix()}`\n",
        f"- **Discoveries CSV**: `{discoveries_csv.as_posix()}`\n",
        "\n## Top 5 Discoveries per Utility Tier\n",
    ]

    for tier, group in grouped:
        tier_str = str(tier)
        top5 = group.head(5)
        if top5.empty:
            continue
        lines.append(f"\n### Tier: {tier_str}\n\n")
        lines.append("| Rank | material_id | formula | predicted_bandgap_eV | discovery_score |\n")
        lines.append("|---:|---|---|---:|---:|\n")
        for i, row in enumerate(top5.itertuples(index=False), start=1):
            lines.append(
                f"| {i} | {getattr(row, 'material_id', '')} | {getattr(row, 'formula', '')} | "
                f"{getattr(row, 'predicted_latent_bandgap_eV', float('nan')):.4f} | "
                f"{getattr(row, 'discovery_score', float('nan')):.4f} |\n"
            )

    summary_md.write_text("".join(lines), encoding="utf-8")
    LOG.warning("Saved discovery visualizations to %s", figures_dir.as_posix())
    LOG.warning("Saved discovery summary to %s", summary_md.as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize DeepESense discovery results.")
    parser.add_argument(
        "--master",
        type=str,
        default=str(Path("data") / "raw" / "deepesense_master_v1.csv"),
        help="Original master CSV.",
    )
    parser.add_argument(
        "--discoveries",
        type=str,
        default=str(Path("results") / "deepesense_discoveries_v1.csv"),
        help="Discoveries CSV.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(Path("reports") / "figures" / "discoveries_v1"),
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--summary-md",
        type=str,
        default=str(Path("reports") / "deepesense_discoveries_summary_v1.md"),
        help="Markdown summary output path.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    visualize_discoveries(
        master_csv=Path(args.master),
        discoveries_csv=Path(args.discoveries),
        figures_dir=Path(args.figures_dir),
        summary_md=Path(args.summary_md),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

