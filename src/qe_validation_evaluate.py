"""
Evaluation — close the evidence loop with QE-recomputed gaps.

Joins the parsed QE results from ``src.qe_validation_parse`` with the Stage-1
and Stage-2 ML predictions (already carried through via the manifest), runs
cheap baselines, and emits the paper figures + Markdown report.

Inputs
------
- ``results/qe_validation_results.csv``  (from qe_validation_parse)

Outputs
-------
- ``results/qe_validation_merged.csv``
- ``reports/figures/qe_validation/*.png``
- ``reports/qe_validation_evaluation.md``

What the report contains
------------------------
1. Validation set composition (counts, status breakdown, selection reasons)
2. Base rate of "actually finite gap when recomputed" — stratified by
   selection reason (the random_baseline slice is the only unbiased estimate)
3. Regression metrics for each scoring method on the finite-gap subset
   (MAE, RMSE, Spearman ρ)
4. Precision @ K table for each scoring method
5. Cheap baselines (random base rate, max-ΔEN heuristic, n_elements heuristic)
6. Disagreement-set sanity: when XGB and GNN disagreed, which one was closer
   to the recomputed gap?
7. Calibration curves
8. Failure inspection: top worst predictions
9. Caveats: conditional precision, PBE-vs-PBE oracle, magnetism, sample size

This file is the evidence backbone for Claims A and B in the paper.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

LOG = logging.getLogger("deepesense.qe_validation_evaluate")

FINITE_GAP_THRESHOLD_eV = 0.05  # what we call "actually a gap" in the truth column

MODEL_COLUMNS: Dict[str, str] = {
    "stage1_xgb_eV": "Stage 1 (XGBoost)",
    "stage2_xgb_eV": "Stage 2 (XGBoost re-eval)",
    "stage2_gnn_eV": "Stage 2 (GNN)",
}


# --- featureless baselines --------------------------------------------------

def _max_en_diff(formula: str) -> float:
    """Max Pauling electronegativity difference across elements in the formula.

    Higher ΔEN ↔ more ionic ↔ more likely to have a gap (Pauling/Goldschmidt).
    A weak but well-motivated baseline.
    """
    try:
        ens: List[float] = []
        for el in Composition(formula).elements:
            x = Element(el.symbol).X
            if x is None or (isinstance(x, float) and math.isnan(x)):
                continue
            ens.append(float(x))
        if len(ens) < 2:
            return 0.0
        return max(ens) - min(ens)
    except Exception:
        return 0.0


def _n_elements(formula: str) -> int:
    try:
        return len({el.symbol for el in Composition(formula).elements})
    except Exception:
        return 0


def add_baseline_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    formulas = out["formula"].astype(str)
    out["baseline_random_score"] = np.random.default_rng(0).random(len(out))
    out["baseline_max_en_diff"] = formulas.apply(_max_en_diff)
    out["baseline_n_elements"] = formulas.apply(_n_elements).astype(float)
    return out


# --- metrics ----------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    # .to_numpy() can hand back a read-only view in newer pandas; force a
    # writable float copy so the in-place centering below is safe.
    ra = np.array(pd.Series(a).rank().to_numpy(), dtype=float, copy=True)
    rb = np.array(pd.Series(b).rank().to_numpy(), dtype=float, copy=True)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return {"n": int(mask.sum()), "mae": float("nan"), "rmse": float("nan"), "spearman": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return {
        "n": int(mask.sum()),
        "mae": mae,
        "rmse": rmse,
        "spearman": _spearman(yt, yp),
    }


def precision_at_k_curve(
    scores: np.ndarray, truth: np.ndarray, ks: List[int]
) -> Dict[int, float]:
    """Of the top-K rows by score, what fraction have truth==True?"""
    out: Dict[int, float] = {}
    mask = ~np.isnan(scores)
    if mask.sum() == 0:
        return {k: float("nan") for k in ks}
    order = np.argsort(-scores[mask])
    truth_sorted = truth[mask][order]
    for k in ks:
        kk = min(k, len(truth_sorted))
        if kk == 0:
            out[k] = float("nan")
        else:
            out[k] = float(truth_sorted[:kk].sum() / kk)
    return out


# --- plotting ---------------------------------------------------------------

def plot_scatter(
    df: pd.DataFrame, model_col: str, truth_col: str, label: str, out_path: Path
) -> None:
    sub = df[[model_col, truth_col]].dropna()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(sub[truth_col], sub[model_col], s=28, alpha=0.75, edgecolor="k", linewidth=0.3)
    lo = float(min(sub[truth_col].min(), sub[model_col].min(), 0.0))
    hi = float(max(sub[truth_col].max(), sub[model_col].max(), 0.5))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="grey")
    ax.set_xlabel("Recomputed PBE gap (eV)")
    ax.set_ylabel(f"{label} predicted gap (eV)")
    metrics = regression_metrics(sub[truth_col].to_numpy(), sub[model_col].to_numpy())
    ax.set_title(
        f"{label}\nMAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
        f"ρ={metrics['spearman']:.2f}  N={metrics['n']}"
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_precision_at_k(
    curves: Dict[str, Dict[int, float]], ks: List[int], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for label, curve in curves.items():
        ys = [curve.get(k, float("nan")) for k in ks]
        ax.plot(ks, ys, marker="o", label=label, linewidth=1.5, markersize=5)
    ax.set_xlabel("K (top-K leads ranked by method)")
    ax.set_ylabel("Precision @ K (truth = recomputed PBE gap > 0.05 eV)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Precision @ K — methods vs cheap baselines")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_calibration(
    df: pd.DataFrame, model_cols: List[str], truth_col: str, out_path: Path
) -> None:
    bins = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
    labels = [f"[{bins[i]:.2f},{bins[i+1]:.2f})" for i in range(len(bins) - 1)]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    width = 0.8 / max(1, len(model_cols))
    centers = np.arange(len(labels))
    for i, mc in enumerate(model_cols):
        sub = df[[mc, truth_col]].dropna()
        if sub.empty:
            continue
        bin_idx = np.digitize(sub[mc].to_numpy(), bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(labels) - 1)
        means = []
        for b in range(len(labels)):
            sel = sub[truth_col].to_numpy()[bin_idx == b]
            means.append(float(np.mean(sel)) if len(sel) > 0 else float("nan"))
        ax.bar(
            centers + (i - (len(model_cols) - 1) / 2) * width,
            means,
            width=width,
            label=MODEL_COLUMNS.get(mc, mc),
            edgecolor="k",
            linewidth=0.3,
        )
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_xlabel("Predicted gap bin (eV)")
    ax.set_ylabel("Mean recomputed PBE gap in bin (eV)")
    ax.set_title("Calibration — predicted bin vs mean true gap")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_disagreement_winner(df: pd.DataFrame, out_path: Path) -> None:
    if "agreement_class" not in df.columns:
        return
    sub = df[
        (df["agreement_class"] == "strong")
        & df["stage2_xgb_eV"].notna()
        & df["stage2_gnn_eV"].notna()
        & df["recomputed_pbe_gap_eV"].notna()
    ].copy()
    if sub.empty:
        return
    xgb_err = (sub["stage2_xgb_eV"] - sub["recomputed_pbe_gap_eV"]).abs()
    gnn_err = (sub["stage2_gnn_eV"] - sub["recomputed_pbe_gap_eV"]).abs()
    xgb_wins = int((xgb_err < gnn_err).sum())
    gnn_wins = int((gnn_err < xgb_err).sum())
    ties = int((xgb_err == gnn_err).sum())

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.bar(["GNN closer", "XGB closer", "tie"], [gnn_wins, xgb_wins, ties], edgecolor="k")
    ax.set_ylabel("Strong-disagreement rows")
    ax.set_title(f"Disagreement-set winner (N={len(sub)})")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# --- markdown report --------------------------------------------------------

def _fmt(v: float, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


def write_markdown_report(
    df: pd.DataFrame,
    metrics: Dict[str, Dict[str, float]],
    pak_curves: Dict[str, Dict[int, float]],
    base_rates: Dict[str, float],
    disagreement_summary: Dict[str, int],
    figures_dir: Path,
    report_path: Path,
    truth_threshold_eV: float,
) -> None:
    lines: List[str] = []
    lines.append("# DeepESense — QE Validation Evaluation\n\n")
    lines.append(
        "> **Claim under evaluation.** A non-trivial fraction of MP entries labelled "
        "`band_gap = 0.0 eV` are actually finite-gap when re-computed with a "
        "rule-fixed `pw.x` protocol; the two-stage ML pipeline concentrates them in "
        "the top-K above cheap baselines.\n\n"
    )

    n_total = len(df)
    n_finite = int((df["recomputed_pbe_gap_eV"] > truth_threshold_eV).fillna(False).sum())
    n_metal = int((df["gap_status"] == "metallic").sum())
    n_ok = int((df["gap_status"] == "ok").sum())
    n_failed = int(df["gap_status"].isin(["unparseable", "no_eigenvalues", "missing_output", "odd_electron_count"]).sum())

    lines.append("## 1. Validation set composition\n\n")
    lines.append(f"- Total candidates parsed: **{n_total}**\n")
    lines.append(f"- Status `ok` (clean finite gap): {n_ok}\n")
    lines.append(f"- Status `metallic` (zero gap): {n_metal}\n")
    lines.append(f"- Status `failed` (unparseable / missing / odd-electron): {n_failed}\n")
    lines.append(f"- **Finite gap above threshold ({truth_threshold_eV:g} eV): {n_finite}**\n\n")

    if "selection_reason" in df.columns:
        lines.append("### Selection reason breakdown\n\n")
        lines.append("| Reason | N | Finite-gap rate |\n|---|---:|---:|\n")
        for reason, sub in df.groupby("selection_reason"):
            n = len(sub)
            finite = int((sub["recomputed_pbe_gap_eV"] > truth_threshold_eV).fillna(False).sum())
            rate = finite / n if n else float("nan")
            lines.append(f"| {reason} | {n} | {_fmt(rate)} |\n")
        lines.append("\n")

    lines.append("## 2. Base rates\n\n")
    for k, v in base_rates.items():
        lines.append(f"- **{k}**: {_fmt(v)}\n")
    lines.append(
        "\n> The `random_baseline` row above is the only **unbiased** estimate of the "
        "underlying base rate; the disagreement / consensus rows are conditional on "
        "the selection rule.\n\n"
    )

    lines.append("## 3. Regression metrics on finite-gap subset\n\n")
    lines.append("Computed only on rows where `gap_status == 'ok'`.\n\n")
    lines.append("| Method | N | MAE (eV) | RMSE (eV) | Spearman ρ |\n|---|---:|---:|---:|---:|\n")
    for label, m in metrics.items():
        lines.append(
            f"| {label} | {m['n']} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | {_fmt(m['spearman'])} |\n"
        )
    lines.append("\n")

    lines.append(f"## 4. Precision @ K (truth = recomputed PBE gap > {truth_threshold_eV:g} eV)\n\n")
    ks = sorted({k for c in pak_curves.values() for k in c.keys()})
    if ks:
        header_ks = " | ".join(f"K={k}" for k in ks)
        lines.append("| Method | " + header_ks + " |\n")
        lines.append("|---|" + "|".join(["---:"] * len(ks)) + "|\n")
        for label, curve in pak_curves.items():
            row = " | ".join(_fmt(curve.get(k, float("nan"))) for k in ks)
            lines.append(f"| {label} | {row} |\n")
        lines.append("\n")
    lines.append(f"![Precision @ K]({(figures_dir / 'precision_at_k.png').as_posix()})\n\n")

    lines.append("## 5. Disagreement-set sanity\n\n")
    lines.append(
        "Of strong-disagreement rows where both Stage-2 models produced a "
        "prediction and the recomputed gap is known:\n\n"
    )
    lines.append(f"- **GNN closer to truth**: {disagreement_summary.get('gnn_wins', 0)}\n")
    lines.append(f"- **XGB closer to truth**: {disagreement_summary.get('xgb_wins', 0)}\n")
    lines.append(f"- Ties: {disagreement_summary.get('ties', 0)}\n")
    lines.append(f"- Total in this slice: {disagreement_summary.get('n', 0)}\n\n")
    lines.append(f"![Disagreement winner]({(figures_dir / 'disagreement_winner.png').as_posix()})\n\n")

    lines.append("## 6. Calibration\n\n")
    lines.append(f"![Calibration]({(figures_dir / 'calibration.png').as_posix()})\n\n")

    lines.append("## 7. Scatter plots (predicted vs recomputed)\n\n")
    for mc, label in MODEL_COLUMNS.items():
        png = figures_dir / f"scatter_{mc}.png"
        if png.exists():
            lines.append(f"### {label}\n\n")
            lines.append(f"![{label}]({png.as_posix()})\n\n")

    lines.append("## 8. Failure inspection — top mispredictions\n\n")
    sub = df[df["gap_status"] == "ok"].copy()
    if not sub.empty:
        sub["abs_err_gnn"] = (sub.get("stage2_gnn_eV") - sub["recomputed_pbe_gap_eV"]).abs()
        worst = sub.dropna(subset=["abs_err_gnn"]).sort_values("abs_err_gnn", ascending=False).head(5)
        if not worst.empty:
            cols = [
                "material_id",
                "formula",
                "selection_reason",
                "recomputed_pbe_gap_eV",
                "stage2_gnn_eV",
                "stage2_xgb_eV",
                "stage1_xgb_eV",
                "abs_err_gnn",
            ]
            cols = [c for c in cols if c in worst.columns]
            lines.append("| " + " | ".join(cols) + " |\n")
            lines.append("|" + "|".join(["---"] * len(cols)) + "|\n")
            for _, r in worst.iterrows():
                vals = []
                for c in cols:
                    v = r[c]
                    vals.append(_fmt(v) if isinstance(v, float) else str(v))
                lines.append("| " + " | ".join(vals) + " |\n")
            lines.append("\n")

    lines.append("## 9. Caveats\n\n")
    lines.append(
        "- **Conditional precision.** Disagreement-driven and consensus-driven\n"
        "  precisions are conditional on the selection rule. Only rows with\n"
        "  `selection_reason = random_baseline` give an unbiased estimate of the\n"
        "  underlying base rate. Quote conditional and unbiased numbers separately\n"
        "  in the paper.\n"
        "- **PBE vs PBE oracle.** The validation oracle is the same theory level as\n"
        "  the training labels. This validates the *database-hygiene* claim ('MP\n"
        "  zero-gap labels are unreliable') but not 'PBE was wrong about the\n"
        "  electronic structure'. HSE / GW would be needed for the latter.\n"
        "- **Magnetism.** The rule-fixed protocol does not enable spin polarization.\n"
        "  Materials with magnetic ground states may be flagged metallic spuriously\n"
        "  and will appear in the failure inspection table above.\n"
        "- **Sample size.** With small N, precision @ K curves have high variance;\n"
        "  bootstrap or pre-register the K values for the paper headline numbers.\n"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")
    LOG.warning("Wrote evaluation report to %s", report_path.as_posix())


# --- top-level orchestration ------------------------------------------------

def evaluate(
    qe_results_csv: Path,
    out_csv: Path,
    figures_dir: Path,
    report_path: Path,
    truth_threshold_eV: float = FINITE_GAP_THRESHOLD_eV,
    ks: Optional[List[int]] = None,
) -> pd.DataFrame:
    if ks is None:
        ks = [1, 3, 5, 10, 15, 20]

    df = pd.read_csv(qe_results_csv)
    if df.empty:
        raise ValueError(f"QE results CSV is empty: {qe_results_csv}")

    df = add_baseline_scores(df)
    truth_mask = (df["recomputed_pbe_gap_eV"] > truth_threshold_eV).fillna(False).to_numpy()
    df["truth_finite_gap"] = truth_mask

    # Regression metrics on the cleanly-parsed finite-gap subset
    finite_subset = df[df["gap_status"] == "ok"].copy()
    metrics: Dict[str, Dict[str, float]] = {}
    for mc, label in MODEL_COLUMNS.items():
        if mc not in finite_subset.columns:
            continue
        m = regression_metrics(
            finite_subset["recomputed_pbe_gap_eV"].to_numpy(),
            pd.to_numeric(finite_subset[mc], errors="coerce").to_numpy(),
        )
        metrics[label] = m

    # Precision @ K curves, on full validated set
    pak_curves: Dict[str, Dict[int, float]] = {}
    score_specs = [
        ("Stage 1 (XGBoost)", "stage1_xgb_eV"),
        ("Stage 2 (XGBoost re-eval)", "stage2_xgb_eV"),
        ("Stage 2 (GNN)", "stage2_gnn_eV"),
        ("Baseline: random", "baseline_random_score"),
        ("Baseline: max ΔEN", "baseline_max_en_diff"),
        ("Baseline: n_elements", "baseline_n_elements"),
    ]
    for label, col in score_specs:
        if col not in df.columns:
            continue
        scores = pd.to_numeric(df[col], errors="coerce").to_numpy()
        pak_curves[label] = precision_at_k_curve(scores, truth_mask.astype(float), ks)

    # Base rates: overall and stratified by selection reason
    base_rates: Dict[str, float] = {"overall_validated_set": float(truth_mask.mean())}
    if "selection_reason" in df.columns:
        for reason, sub in df.groupby("selection_reason"):
            base_rates[str(reason)] = float(
                (sub["recomputed_pbe_gap_eV"] > truth_threshold_eV).fillna(False).mean()
            )

    # Disagreement-set sanity numbers (also rendered as a plot)
    disagreement_summary = {"n": 0, "gnn_wins": 0, "xgb_wins": 0, "ties": 0}
    if "agreement_class" in df.columns:
        d = df[
            (df["agreement_class"] == "strong")
            & df["stage2_xgb_eV"].notna()
            & df["stage2_gnn_eV"].notna()
            & df["recomputed_pbe_gap_eV"].notna()
        ].copy()
        if not d.empty:
            xgb_err = (d["stage2_xgb_eV"] - d["recomputed_pbe_gap_eV"]).abs()
            gnn_err = (d["stage2_gnn_eV"] - d["recomputed_pbe_gap_eV"]).abs()
            disagreement_summary["n"] = len(d)
            disagreement_summary["gnn_wins"] = int((gnn_err < xgb_err).sum())
            disagreement_summary["xgb_wins"] = int((xgb_err < gnn_err).sum())
            disagreement_summary["ties"] = int((xgb_err == gnn_err).sum())

    # Plots
    figures_dir.mkdir(parents=True, exist_ok=True)
    for mc in MODEL_COLUMNS:
        if mc in df.columns:
            plot_scatter(
                df, mc, "recomputed_pbe_gap_eV", MODEL_COLUMNS[mc], figures_dir / f"scatter_{mc}.png"
            )
    plot_precision_at_k(pak_curves, ks, figures_dir / "precision_at_k.png")
    plot_calibration(
        df, [c for c in MODEL_COLUMNS if c in df.columns], "recomputed_pbe_gap_eV", figures_dir / "calibration.png"
    )
    plot_disagreement_winner(df, figures_dir / "disagreement_winner.png")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary_meta = {
        "qe_results_csv": str(qe_results_csv),
        "n_validated": int(len(df)),
        "n_finite_truth": int(truth_mask.sum()),
        "truth_threshold_eV": float(truth_threshold_eV),
        "ks_evaluated": ks,
        "base_rates": base_rates,
        "regression_metrics": metrics,
        "precision_at_k": {k: {kk: vv for kk, vv in v.items()} for k, v in pak_curves.items()},
        "disagreement_summary": disagreement_summary,
    }
    out_csv.with_suffix(out_csv.suffix + ".meta.json").write_text(
        json.dumps(summary_meta, indent=2), encoding="utf-8"
    )

    write_markdown_report(
        df=df,
        metrics=metrics,
        pak_curves=pak_curves,
        base_rates=base_rates,
        disagreement_summary=disagreement_summary,
        figures_dir=figures_dir,
        report_path=report_path,
        truth_threshold_eV=truth_threshold_eV,
    )
    return df


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate ML predictions against QE-recomputed gaps.")
    p.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=str(Path("results") / "qe_validation_results.csv"),
        help="Parsed QE results CSV from qe_validation_parse.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(Path("results") / "qe_validation_merged.csv"),
        help="Output merged CSV (validation rows × ML predictions × baselines).",
    )
    p.add_argument(
        "--figures-dir",
        type=str,
        default=str(Path("reports") / "figures" / "qe_validation"),
    )
    p.add_argument(
        "--report",
        type=str,
        default=str(Path("reports") / "qe_validation_evaluation.md"),
    )
    p.add_argument(
        "--truth-threshold",
        type=float,
        default=FINITE_GAP_THRESHOLD_eV,
        help="Recomputed PBE gap above this counts as 'truth = finite gap' for precision@K.",
    )
    p.add_argument(
        "--ks",
        type=str,
        default="1,3,5,10,15,20",
        help="Comma-separated K values for precision@K.",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    _configure_logging(args.verbose)

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    evaluate(
        qe_results_csv=Path(args.inp),
        out_csv=Path(args.out),
        figures_dir=Path(args.figures_dir),
        report_path=Path(args.report),
        truth_threshold_eV=float(args.truth_threshold),
        ks=ks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
