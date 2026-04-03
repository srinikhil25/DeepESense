from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
from xgboost import XGBRegressor


LOG = logging.getLogger("deepesense.discovery_inference")


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


def _feature_columns(df: pd.DataFrame, bg_col: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    drop_cols = {bg_col}
    return [c for c in numeric_cols if c not in drop_cols]


def _build_discovery_score(df: pd.DataFrame) -> pd.Series:
    # Higher predicted bandgap is better; lower ehull is better.
    if "energy_above_hull" in df.columns:
        ehull = pd.to_numeric(df["energy_above_hull"], errors="coerce").fillna(0.05)
    elif "energy_above_hull_eV" in df.columns:
        ehull = pd.to_numeric(df["energy_above_hull_eV"], errors="coerce").fillna(0.05)
    else:
        ehull = pd.Series([0.0] * len(df), index=df.index)

    # Clamp to [0, 0.05], then convert to stability factor in [0,1].
    ehull_clamped = ehull.clip(lower=0.0, upper=0.05)
    stability_factor = 1.0 - (ehull_clamped / 0.05)
    return df["predicted_latent_bandgap_eV"] * (0.7 + 0.3 * stability_factor)


def run_inference(
    model_path: Path,
    features_csv: Path,
    out_csv: Path,
    min_pred_bandgap: float = 0.5,
    top_k: int = 100,
) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    if df.empty:
        raise ValueError(f"Input features CSV is empty: {features_csv}")

    bg_col = _select_bandgap_column(df)
    df[bg_col] = pd.to_numeric(df[bg_col], errors="coerce")

    if "is_dark_matter" in df.columns:
        dark = df[df["is_dark_matter"] == True].copy()  # noqa: E712
    else:
        dark = df[df[bg_col] == 0].copy()
        dark["is_dark_matter"] = True

    if dark.empty:
        raise ValueError("No dark-matter rows found for inference.")

    feature_cols = _feature_columns(df, bg_col)
    if not feature_cols:
        raise ValueError("No numeric feature columns found for model inference.")

    x_dark = dark[feature_cols].copy()
    x_dark = x_dark.fillna(x_dark.median(numeric_only=True))

    model = XGBRegressor()
    model.load_model(str(model_path))
    dark["predicted_latent_bandgap_eV"] = model.predict(x_dark)

    candidates = dark[dark["predicted_latent_bandgap_eV"] > float(min_pred_bandgap)].copy()
    if candidates.empty:
        LOG.warning("No candidates passed min_pred_bandgap > %.3f eV.", min_pred_bandgap)

    candidates["discovery_score"] = _build_discovery_score(candidates)
    ranked = candidates.sort_values("discovery_score", ascending=False).head(int(top_k)).copy()

    if "formula" not in ranked.columns:
        ranked["formula"] = pd.NA
    if "utility_tier" not in ranked.columns:
        ranked["utility_tier"] = pd.NA

    out_cols = [
        "material_id",
        "formula",
        "utility_tier",
        "predicted_latent_bandgap_eV",
        "discovery_score",
        "energy_above_hull" if "energy_above_hull" in ranked.columns else "energy_above_hull_eV",
        "is_dark_matter",
    ]
    out_cols = [c for c in out_cols if c in ranked.columns]
    out = ranked[out_cols].copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    LOG.warning("Saved %d ranked discoveries to %s", len(out), out_csv.as_posix())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer latent bandgaps for DeepESense dark matter materials.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path("models") / "baseline_xgboost_v1.json"),
        help="Trained XGBoost model path.",
    )
    parser.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=str(Path("data") / "processed" / "deepesense_features_v1.csv"),
        help="Input feature matrix CSV.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("results") / "deepesense_discoveries_v1.csv"),
        help="Output discoveries CSV path.",
    )
    parser.add_argument("--min-pred-bandgap", type=float, default=0.5, help="Minimum predicted latent bandgap.")
    parser.add_argument("--top-k", type=int, default=100, help="Number of top discoveries to save.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    run_inference(
        model_path=Path(args.model),
        features_csv=Path(args.inp),
        out_csv=Path(args.out),
        min_pred_bandgap=float(args.min_pred_bandgap),
        top_k=int(args.top_k),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

