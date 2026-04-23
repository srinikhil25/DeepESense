"""
Stage 2 — Structure-aware GNN fidelity refinement.

Takes the Stage-1 unfolded dark-matter leads from
``results/deepesense_dark_matter_unfolded.csv`` (or any compatible CSV with
a ``material_id`` and a Stage-1 score column), pulls the corresponding crystal
structures out of the master corpus, and re-predicts each band gap with the
trained CGCNN-style GNN.

For each lead the script reports:
- ``stage1_xgb_eV``  the original Stage-1 prediction (carried forward)
- ``stage2_xgb_eV``  XGBoost re-evaluated using train medians from the model
                     meta sidecar (so it is byte-consistent with training)
- ``stage2_gnn_eV``  GNN prediction on the actual crystal structure
- ``disagreement_eV`` |stage2_gnn - stage2_xgb|, magnitude of model disagreement
- ``agreement_class`` one of {agree, mild, strong} for downstream filtering
- ``finite_gap_consensus``  True iff both models predict > tol

The disagreement column is the primary QE-validation trigger: rows where the
two models disagree strongly are the most physically informative to validate
with first-principles, because at least one of the two ML answers must be
wrong.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import torch
from pymatgen.core import Structure
from xgboost import XGBRegressor

from .train_baseline import _select_bandgap_column, _select_feature_columns
from .train_gnn import load_gnn_checkpoint, predict_bandgap_gnn

LOG = logging.getLogger("deepesense.compare_gnn_xgboost")

AGREE_TOL_eV = 0.30
STRONG_DISAGREE_eV = 1.00
FINITE_GAP_TOL_eV = 0.10


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _classify_agreement(delta: float) -> str:
    if delta != delta:  # NaN
        return "unknown"
    a = abs(delta)
    if a <= AGREE_TOL_eV:
        return "agree"
    if a >= STRONG_DISAGREE_eV:
        return "strong"
    return "mild"


def _load_meta_sidecar(model_path: Path) -> Optional[dict]:
    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    if not meta_path.exists():
        LOG.warning(
            "No XGB meta sidecar at %s — Stage-2 XGB inference will fall back to "
            "live-batch medians and will NOT match training. Retrain via train_baseline.",
            meta_path.as_posix(),
        )
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_structures_for_ids(master_csv: Path, ids: Set[str], chunksize: int = 3000) -> Dict[str, Structure]:
    out: Dict[str, Structure] = {}
    reader = pd.read_csv(master_csv, usecols=["material_id", "structure"], chunksize=chunksize)
    for chunk in reader:
        sub = chunk[chunk["material_id"].astype(str).isin(ids)]
        for _, row in sub.iterrows():
            mid = str(row["material_id"])
            if mid in out:
                continue
            raw = row["structure"]
            if pd.isna(raw):
                continue
            try:
                d = json.loads(raw) if isinstance(raw, str) else raw
                out[mid] = Structure.from_dict(d)
            except Exception:
                continue
        if len(out) >= len(ids):
            break
    return out


def _detect_stage1_score_col(disc: pd.DataFrame) -> str:
    for c in ("stage1_score", "discovery_score"):
        if c in disc.columns:
            return c
    raise ValueError("Stage-1 input CSV needs a 'stage1_score' or legacy 'discovery_score' column.")


def _detect_stage1_pred_col(disc: pd.DataFrame) -> Optional[str]:
    for c in ("predicted_latent_bandgap_eV", "stage1_xgb_eV"):
        if c in disc.columns:
            return c
    return None


def refine_with_gnn(
    discoveries_csv: Path,
    master_csv: Path,
    features_csv: Path,
    gnn_ckpt: Path,
    xgb_model_path: Path,
    out_csv: Path,
    top_k: int,
    device: torch.device,
) -> pd.DataFrame:
    disc = pd.read_csv(discoveries_csv)
    if disc.empty:
        raise ValueError(f"Stage-1 CSV is empty: {discoveries_csv}")

    score_col = _detect_stage1_score_col(disc)
    pred_col = _detect_stage1_pred_col(disc)
    disc = disc.sort_values(score_col, ascending=False).head(int(top_k)).reset_index(drop=True)
    ids: Set[str] = set(disc["material_id"].astype(str))

    LOG.info("Loading GNN checkpoint from %s", gnn_ckpt)
    model_gnn, cfg = load_gnn_checkpoint(gnn_ckpt, device)
    cutoff = float(cfg["cutoff"])
    num_gaussians = int(cfg["num_gaussians"])
    max_neighbors = int(cfg.get("max_num_neighbors", 32))

    LOG.info("Pulling %d structures from master corpus", len(ids))
    structs = load_structures_for_ids(master_csv, ids)

    # XGBoost: use the SAME train medians + feature schema that produced the
    # original Stage-1 ranking. Falling back to live-batch medians is what
    # caused the previous "comparison" plot to be misleading.
    df_feat = pd.read_csv(features_csv)
    bg_col = _select_bandgap_column(df_feat)
    meta = _load_meta_sidecar(xgb_model_path)
    if meta is not None:
        feat_cols: List[str] = list(meta["feature_cols"])
        train_medians = pd.Series(meta["train_medians"], dtype="float64")
    else:
        feat_cols = _select_feature_columns(df_feat, bg_col)
        train_medians = df_feat[feat_cols].median(numeric_only=True)

    df_sub = df_feat[df_feat["material_id"].astype(str).isin(ids)].copy()
    for c in feat_cols:
        if c not in df_sub.columns:
            df_sub[c] = train_medians.get(c)

    xgb = XGBRegressor()
    xgb.load_model(str(xgb_model_path))

    rows: List[dict] = []
    for _, r in disc.iterrows():
        mid = str(r["material_id"])

        gnn_pred = float("nan")
        if mid in structs:
            gnn_pred = predict_bandgap_gnn(
                structs[mid], model_gnn, device, cutoff, num_gaussians, max_neighbors
            )

        xgb_pred = float("nan")
        mrow = df_sub[df_sub["material_id"].astype(str) == mid]
        if not mrow.empty:
            x = mrow[feat_cols].fillna(train_medians).iloc[0:1]
            xgb_pred = float(xgb.predict(x.values)[0])

        delta = (gnn_pred - xgb_pred) if (gnn_pred == gnn_pred and xgb_pred == xgb_pred) else float("nan")
        finite_consensus = (
            (gnn_pred == gnn_pred and xgb_pred == xgb_pred)
            and gnn_pred > FINITE_GAP_TOL_eV
            and xgb_pred > FINITE_GAP_TOL_eV
        )

        rows.append(
            {
                "material_id": mid,
                "formula": r.get("formula", ""),
                "utility_tier": r.get("utility_tier", ""),
                "stage1_score": float(r[score_col]),
                "stage1_xgb_eV": float(r[pred_col]) if pred_col else float("nan"),
                "stage2_xgb_eV": xgb_pred,
                "stage2_gnn_eV": gnn_pred,
                "disagreement_eV": delta,
                "agreement_class": _classify_agreement(delta),
                "finite_gap_consensus": bool(finite_consensus),
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    sidecar = {
        "stage": 2,
        "stage1_csv": str(discoveries_csv),
        "gnn_ckpt": str(gnn_ckpt),
        "xgb_model": str(xgb_model_path),
        "top_k": int(top_k),
        "agree_tol_eV": AGREE_TOL_eV,
        "strong_disagree_eV": STRONG_DISAGREE_eV,
        "finite_gap_tol_eV": FINITE_GAP_TOL_eV,
        "xgb_imputation_source": "model_meta_sidecar" if meta is not None else "live_batch_fallback",
        "n_with_gnn": int(out_df["stage2_gnn_eV"].notna().sum()),
        "n_with_xgb": int(out_df["stage2_xgb_eV"].notna().sum()),
        "n_strong_disagreement": int((out_df["agreement_class"] == "strong").sum()),
        "n_finite_gap_consensus": int(out_df["finite_gap_consensus"].sum()),
    }
    out_csv.with_suffix(out_csv.suffix + ".meta.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    LOG.warning("Stage 2 wrote %d refined rows to %s", len(out_df), out_csv.as_posix())
    return out_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2: GNN fidelity refinement of Stage-1 dark-matter leads.")
    parser.add_argument(
        "--discoveries",
        type=str,
        default=str(Path("results") / "deepesense_dark_matter_unfolded.csv"),
        help="Stage-1 unfolded dark-matter CSV.",
    )
    parser.add_argument(
        "--master",
        type=str,
        default=str(Path("data") / "raw" / "deepesense_candidates.csv"),
    )
    parser.add_argument(
        "--features",
        type=str,
        default=str(Path("data") / "processed" / "deepesense_features.csv"),
    )
    parser.add_argument(
        "--gnn-ckpt",
        type=str,
        default=str(Path("models") / "cgcnn.pt"),
    )
    parser.add_argument(
        "--xgb-model",
        type=str,
        default=str(Path("models") / "baseline_xgboost.json"),
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("results") / "deepesense_stage2_refined.csv"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()
    _configure_logging(args.verbose)

    refine_with_gnn(
        discoveries_csv=Path(args.discoveries),
        master_csv=Path(args.master),
        features_csv=Path(args.features),
        gnn_ckpt=Path(args.gnn_ckpt),
        xgb_model_path=Path(args.xgb_model),
        out_csv=Path(args.out),
        top_k=int(args.top_k),
        device=torch.device(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
