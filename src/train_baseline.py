from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


LOG = logging.getLogger("deepesense.train_baseline")


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
    raise ValueError("No bandgap column found. Expected 'bandgap_eV' or 'band_gap'.")


def _select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    drop = {target_col}
    return [c for c in numeric_cols if c not in drop]


def train_baseline(inp_csv: Path, model_out: Path, report_out: Path, parity_plot_out: Path) -> None:
    df = pd.read_csv(inp_csv)
    if df.empty:
        raise ValueError(f"Input is empty: {inp_csv}")

    bg_col = _select_bandgap_column(df)
    df[bg_col] = pd.to_numeric(df[bg_col], errors="coerce")
    labeled = df[df[bg_col] > 0].copy()
    if labeled.empty:
        raise ValueError("No labeled rows found (bandgap > 0).")

    feature_cols = _select_feature_columns(labeled, bg_col)
    if not feature_cols:
        raise ValueError("No numeric feature columns available for model training.")

    x = labeled[feature_cols].copy()
    x = x.fillna(x.median(numeric_only=True))
    y = labeled[bg_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    LOG.info("Training rows: %d | Testing rows: %d", len(x_train), len(x_test))

    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_out))

    parity_plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(y_test, y_pred, s=10, alpha=0.5)
    lo = min(float(y_test.min()), float(y_pred.min()))
    hi = max(float(y_test.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Actual Bandgap (eV)")
    ax.set_ylabel("Predicted Bandgap (eV)")
    ax.set_title("Parity Plot: Predicted vs Actual Bandgap")
    fig.tight_layout()
    fig.savefig(parity_plot_out, dpi=180)
    plt.close(fig)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "DeepESense Baseline Training Report\n",
        f"input_csv: {inp_csv.as_posix()}\n",
        f"labeled_rows: {len(labeled)}\n",
        f"train_rows: {len(x_train)}\n",
        f"test_rows: {len(x_test)}\n",
        f"num_features: {len(feature_cols)}\n",
        f"MAE: {mae:.6f}\n",
        f"R2: {r2:.6f}\n",
        f"model_path: {model_out.as_posix()}\n",
        f"parity_plot: {parity_plot_out.as_posix()}\n",
    ]
    report_out.write_text("".join(report_lines), encoding="utf-8")

    LOG.warning("Training complete. MAE=%.6f, R2=%.6f", mae, r2)
    LOG.warning("Model saved to %s", model_out.as_posix())
    LOG.warning("Parity plot saved to %s", parity_plot_out.as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline XGBoost regressor for bandgap prediction.")
    parser.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=str(Path("data") / "processed" / "deepesense_features_v1.csv"),
        help="Input featurized CSV.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=str(Path("models") / "baseline_xgboost_v1.json"),
        help="Output model path.",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=str(Path("logs") / "baseline_training_summary.txt"),
        help="Training report output path.",
    )
    parser.add_argument(
        "--parity-plot-out",
        type=str,
        default=str(Path("reports") / "figures" / "baseline_v1" / "parity_plot.png"),
        help="Parity plot image output path.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    train_baseline(
        inp_csv=Path(args.inp),
        model_out=Path(args.model_out),
        report_out=Path(args.report_out),
        parity_plot_out=Path(args.parity_plot_out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

