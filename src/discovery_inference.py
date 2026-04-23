"""
Stage 1 — Electronic dark matter unfolding (scale).

Takes the featurized master corpus, isolates the "electronic dark matter"
subset (Materials Project rows whose PBE band gap is reported as effectively
zero), and uses the trained tabular XGBoost regressor to score how plausibly
each one is actually a finite-gap material that PBE collapsed.

This stage is intentionally cheap and high-recall. It does NOT make a
thermoelectric or wearable claim — that is a downstream mission filter
(`src/te_mission_prioritize.py`). The score here is just a calibrated
re-ranking of "dark matter" candidates by predicted latent gap, with a soft
stability tiebreaker.

Outputs
-------
- ``results/deepesense_dark_matter_unfolded.csv`` — ranked Stage-1 leads
- ``results/deepesense_dark_matter_unfolded.meta.json`` — provenance:
  model path, feature schema, dark-matter tolerance, row counts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


LOG = logging.getLogger("deepesense.discovery_inference")

DARK_MATTER_TOL_eV = 1e-3  # treat |Eg| < 1 meV as PBE "zero gap"


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


def _load_meta_sidecar(model_path: Path) -> Optional[dict]:
    """Load `<model>.meta.json` if it exists. Train_baseline writes it."""
    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    if not meta_path.exists():
        LOG.warning(
            "No meta sidecar at %s — falling back to dataframe-derived feature columns "
            "and dark-matter median imputation. This is NOT consistent with training; "
            "retrain via train_baseline.py to produce the sidecar.",
            meta_path.as_posix(),
        )
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _stability_tiebreak(df: pd.DataFrame, ehull_max: float = 0.05) -> pd.Series:
    """Smooth [0,1] factor that gently rewards more-stable rows. Tiebreaker only."""
    for col in ("energy_above_hull", "energy_above_hull_eV"):
        if col in df.columns:
            ehull = pd.to_numeric(df[col], errors="coerce").fillna(ehull_max)
            break
    else:
        return pd.Series(1.0, index=df.index)
    ehull_clamped = ehull.clip(lower=0.0, upper=ehull_max)
    return 1.0 - (ehull_clamped / ehull_max)


def _select_dark_matter(df: pd.DataFrame, bg_col: str, tol_eV: float) -> pd.DataFrame:
    bg = pd.to_numeric(df[bg_col], errors="coerce")
    mask = bg.abs() < tol_eV
    if "is_dark_matter" in df.columns:
        flagged = df["is_dark_matter"].astype("boolean").fillna(False)
        mask = mask | flagged
    out = df.loc[mask].copy()
    out["is_dark_matter"] = True
    return out


def unfold_dark_matter(
    model_path: Path,
    features_csv: Path,
    out_csv: Path,
    min_pred_bandgap_eV: float = 0.10,
    top_k: Optional[int] = None,
    dark_matter_tol_eV: float = DARK_MATTER_TOL_eV,
) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    if df.empty:
        raise ValueError(f"Input features CSV is empty: {features_csv}")

    bg_col = _select_bandgap_column(df)
    df[bg_col] = pd.to_numeric(df[bg_col], errors="coerce")

    dark = _select_dark_matter(df, bg_col, dark_matter_tol_eV)
    if dark.empty:
        raise ValueError(
            f"No dark-matter rows found at tol={dark_matter_tol_eV:g} eV; "
            f"check that {features_csv} preserves PBE-zero entries."
        )
    LOG.info("Dark-matter pool: %d rows (tol=%.1e eV)", len(dark), dark_matter_tol_eV)

    meta = _load_meta_sidecar(model_path)
    if meta is not None:
        feature_cols: List[str] = list(meta["feature_cols"])
        train_medians = pd.Series(meta["train_medians"], dtype="float64")
        # Defensive: any train feature missing in the live dataframe gets filled with its train median.
        missing = [c for c in feature_cols if c not in dark.columns]
        for c in missing:
            dark[c] = train_medians.get(c, np.nan)
        if missing:
            LOG.warning("Filled %d missing feature columns with train medians: %s", len(missing), missing[:6])
    else:
        numeric_cols = dark.select_dtypes(include=["number"]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != bg_col]
        train_medians = dark[feature_cols].median(numeric_only=True)
        if not feature_cols:
            raise ValueError("No numeric feature columns found for model inference.")

    x_dark = dark[feature_cols].copy()
    x_dark = x_dark.fillna(train_medians)

    model = XGBRegressor()
    model.load_model(str(model_path))
    dark["predicted_latent_bandgap_eV"] = model.predict(x_dark.values)

    # Stage-1 score: predicted latent gap, soft-weighted by hull stability.
    # NB: this is a re-ranking of the dark-matter pool, NOT a thermoelectric
    # figure of merit. Higher score == more confidently a finite-gap material.
    stability = _stability_tiebreak(dark)
    dark["stage1_score"] = dark["predicted_latent_bandgap_eV"] * (0.85 + 0.15 * stability)

    candidates = dark[dark["predicted_latent_bandgap_eV"] > float(min_pred_bandgap_eV)].copy()
    if candidates.empty:
        LOG.warning(
            "No candidates passed min_pred_bandgap > %.3f eV (out of %d dark-matter rows).",
            min_pred_bandgap_eV,
            len(dark),
        )
    ranked = candidates.sort_values("stage1_score", ascending=False)
    if top_k is not None and top_k > 0:
        ranked = ranked.head(int(top_k))

    for col in ("formula", "utility_tier"):
        if col not in ranked.columns:
            ranked[col] = pd.NA

    out_cols = [
        "material_id",
        "formula",
        "utility_tier",
        "predicted_latent_bandgap_eV",
        "stage1_score",
        "energy_above_hull" if "energy_above_hull" in ranked.columns else "energy_above_hull_eV",
        "is_dark_matter",
    ]
    out_cols = [c for c in out_cols if c in ranked.columns]
    out = ranked[out_cols].copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    sidecar = {
        "stage": 1,
        "model_path": str(model_path),
        "features_csv": str(features_csv),
        "dark_matter_tol_eV": float(dark_matter_tol_eV),
        "min_pred_bandgap_eV": float(min_pred_bandgap_eV),
        "n_dark_matter_pool": int(len(dark)),
        "n_passed_min_gap": int(len(candidates)),
        "n_written": int(len(out)),
        "feature_cols_used": feature_cols,
        "train_medians_source": "model_meta_sidecar" if meta is not None else "dark_matter_pool_fallback",
        "score_formula": "predicted_latent_bandgap_eV * (0.85 + 0.15 * stability_factor)",
    }
    out_meta = out_csv.with_suffix(out_csv.suffix + ".meta.json")
    out_meta.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    LOG.warning("Stage 1 unfolded %d dark-matter leads to %s", len(out), out_csv.as_posix())
    LOG.warning("Stage 1 provenance: %s", out_meta.as_posix())
    return out


# Back-compat alias: older notebooks / scripts may import run_inference.
run_inference = unfold_dark_matter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1: unfold electronic dark matter with the tabular XGBoost model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path("models") / "baseline_xgboost.json"),
        help="Trained XGBoost model path (a `.meta.json` sidecar is loaded if present).",
    )
    parser.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=str(Path("data") / "processed" / "deepesense_features.csv"),
        help="Input feature matrix CSV.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("results") / "deepesense_dark_matter_unfolded.csv"),
        help="Output Stage-1 ranked CSV path.",
    )
    parser.add_argument(
        "--min-pred-bandgap",
        type=float,
        default=0.10,
        help="Minimum predicted latent bandgap to keep (eV). Default 0.10 keeps the narrow-gap regime in scope.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional cap on rows written. Omit to keep the full ranked list.",
    )
    parser.add_argument(
        "--dark-matter-tol",
        type=float,
        default=DARK_MATTER_TOL_eV,
        help="Treat |band_gap| below this many eV as PBE 'zero gap' (default 1e-3).",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    unfold_dark_matter(
        model_path=Path(args.model),
        features_csv=Path(args.inp),
        out_csv=Path(args.out),
        min_pred_bandgap_eV=float(args.min_pred_bandgap),
        top_k=args.top_k,
        dark_matter_tol_eV=float(args.dark_matter_tol),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
