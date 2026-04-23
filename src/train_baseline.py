from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from .holdout_split import (
    DEFAULT_SPLIT_PATH,
    TEST,
    TRAIN,
    VAL,
    bucket_series,
    load_or_create_split,
)

LOG = logging.getLogger("deepesense.train_baseline")

# Columns that must never be used as features (label leakage or metadata).
NON_FEATURE_COLS = {"material_id", "formula", "pretty_formula", "formula_pretty",
                    "utility_tier", "bandgap_eV", "band_gap"}


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
    drop = set(NON_FEATURE_COLS) | {target_col}
    return [c for c in numeric_cols if c not in drop]


def train_baseline(
    inp_csv: Path,
    model_out: Path,
    report_out: Path,
    parity_plot_out: Path,
    split_path: Path = DEFAULT_SPLIT_PATH,
) -> None:
    df = pd.read_csv(inp_csv)
    if df.empty:
        raise ValueError(f"Input is empty: {inp_csv}")
    if "material_id" not in df.columns:
        raise ValueError("Input CSV must carry a material_id column.")
    if "formula" not in df.columns and "pretty_formula" not in df.columns:
        raise ValueError("Input CSV must carry a formula column for the chemsys holdout split.")

    bg_col = _select_bandgap_column(df)
    df[bg_col] = pd.to_numeric(df[bg_col], errors="coerce")
    labeled = df[df[bg_col] > 0].copy().reset_index(drop=True)
    if labeled.empty:
        raise ValueError("No labeled rows found (bandgap > 0).")

    # ---- Build/load the held-out-by-chemistry split ------------------------
    split = load_or_create_split(labeled, path=split_path)
    buckets = bucket_series(labeled, split)
    train_mask = (buckets == TRAIN).values
    val_mask = (buckets == VAL).values
    test_mask = (buckets == TEST).values
    LOG.warning(
        "Holdout split sizes (rows): train=%d, val=%d, test=%d (out of %d labeled)",
        int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()), len(labeled),
    )
    if train_mask.sum() == 0:
        raise ValueError("Holdout split produced 0 train rows — check seed/fractions.")
    if val_mask.sum() == 0:
        raise ValueError("Holdout split produced 0 val rows — refuse to train without an early-stopping set.")
    if test_mask.sum() == 0:
        LOG.warning("Holdout split produced 0 test rows — test metrics will be NaN.")

    feature_cols = _select_feature_columns(labeled, bg_col)
    if not feature_cols:
        raise ValueError("No numeric feature columns available for model training.")

    x_raw = labeled[feature_cols]
    y = labeled[bg_col].astype(float)

    # Impute with TRAIN medians only — no val/test leakage into the imputer.
    train_medians = x_raw.loc[train_mask].median(numeric_only=True)
    x = x_raw.fillna(train_medians)

    x_train, y_train = x.loc[train_mask], y.loc[train_mask]
    x_val, y_val = x.loc[val_mask], y.loc[val_mask]
    x_test, y_test = x.loc[test_mask], y.loc[test_mask]

    model = XGBRegressor(
        n_estimators=1500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="mae",
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    best_iter = int(getattr(model, "best_iteration", model.n_estimators) or 0)

    # Val metrics (used for early stopping).
    y_val_pred = model.predict(x_val)
    val_mae = float(mean_absolute_error(y_val, y_val_pred))
    val_r2 = float(r2_score(y_val, y_val_pred))

    # Test metrics (NEVER seen during training or early stopping).
    if len(x_test) > 0:
        y_test_pred = model.predict(x_test)
        test_mae = float(mean_absolute_error(y_test, y_test_pred))
        test_r2 = float(r2_score(y_test, y_test_pred))
    else:
        y_test_pred = None
        test_mae = float("nan")
        test_r2 = float("nan")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_out))

    meta_path = model_out.with_suffix(model_out.suffix + ".meta.json")
    meta = {
        "target_col": bg_col,
        "feature_cols": feature_cols,
        "train_medians": {k: (None if pd.isna(v) else float(v)) for k, v in train_medians.items()},
        "split_path": str(split_path),
        "split_method": split.get("method", "by_chemsys_hash"),
        "split_seed": int(split.get("seed", 42)),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "best_iteration": best_iter,
        "val_mae": val_mae,
        "val_r2": val_r2,
        "test_mae": test_mae,
        "test_r2": test_r2,
        # Back-compat shim: older consumers read `mae`/`r2`; point them at test.
        "mae": test_mae,
        "r2": test_r2,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG.warning("Wrote model meta sidecar to %s", meta_path.as_posix())

    # Parity plot on the held-out test bucket (or val if test empty).
    parity_plot_out.parent.mkdir(parents=True, exist_ok=True)
    if y_test_pred is not None and len(y_test) > 0:
        y_plot_true, y_plot_pred, plot_label = y_test, y_test_pred, "Held-out TEST bucket"
    else:
        y_plot_true, y_plot_pred, plot_label = y_val, y_val_pred, "VAL bucket (test empty)"
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(y_plot_true, y_plot_pred, s=10, alpha=0.5)
    lo = min(float(y_plot_true.min()), float(y_plot_pred.min()))
    hi = max(float(y_plot_true.max()), float(y_plot_pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Actual Bandgap (eV)")
    ax.set_ylabel("Predicted Bandgap (eV)")
    ax.set_title(f"Parity Plot — {plot_label}")
    fig.tight_layout()
    fig.savefig(parity_plot_out, dpi=180)
    plt.close(fig)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "DeepESense Baseline Training Report\n",
        f"input_csv: {inp_csv.as_posix()}\n",
        f"labeled_rows: {len(labeled)}\n",
        f"split_method: {split.get('method')}\n",
        f"split_seed: {split.get('seed')}\n",
        f"train_rows: {int(train_mask.sum())}\n",
        f"val_rows: {int(val_mask.sum())}\n",
        f"test_rows: {int(test_mask.sum())}\n",
        f"num_features: {len(feature_cols)}\n",
        f"best_iteration: {best_iter}\n",
        f"val_MAE: {val_mae:.6f}\n",
        f"val_R2: {val_r2:.6f}\n",
        f"test_MAE: {test_mae:.6f}\n",
        f"test_R2: {test_r2:.6f}\n",
        f"model_path: {model_out.as_posix()}\n",
        f"parity_plot: {parity_plot_out.as_posix()}\n",
    ]
    report_out.write_text("".join(report_lines), encoding="utf-8")

    LOG.warning(
        "Training complete. best_iter=%d | val_MAE=%.5f | test_MAE=%.5f | test_R2=%.4f",
        best_iter, val_mae, test_mae, test_r2,
    )
    LOG.warning("Model saved to %s", model_out.as_posix())
    LOG.warning("Parity plot saved to %s", parity_plot_out.as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline XGBoost regressor for bandgap prediction.")
    parser.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=str(Path("data") / "processed" / "deepesense_features.csv"),
        help="Input featurized CSV.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=str(Path("models") / "baseline_xgboost.json"),
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
    parser.add_argument(
        "--split-path",
        type=str,
        default=str(DEFAULT_SPLIT_PATH),
        help="Persisted chemsys holdout split JSON (created if missing).",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    train_baseline(
        inp_csv=Path(args.inp),
        model_out=Path(args.model_out),
        report_out=Path(args.report_out),
        parity_plot_out=Path(args.parity_plot_out),
        split_path=Path(args.split_path),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
