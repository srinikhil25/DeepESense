from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Set

import pandas as pd
import torch
from pymatgen.core import Structure
from xgboost import XGBRegressor

from .train_baseline import _select_bandgap_column, _select_feature_columns
from .train_gnn import load_gnn_checkpoint, predict_bandgap_gnn

LOG = logging.getLogger("deepesense.compare_gnn_xgboost")


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def load_structures_for_ids(master_csv: Path, ids: Set[str], chunksize: int = 3000) -> dict[str, Structure]:
    out: dict[str, Structure] = {}
    sample = pd.read_csv(master_csv, nrows=1)
    need_cols = ["material_id", "structure"]
    reader = pd.read_csv(master_csv, usecols=need_cols, chunksize=chunksize)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare GNN vs XGBoost on top dark-matter discoveries.")
    parser.add_argument(
        "--discoveries",
        type=str,
        default=str(Path("results") / "deepesense_discoveries_v1.csv"),
    )
    parser.add_argument(
        "--master",
        type=str,
        default=str(Path("data") / "raw" / "deepesense_master_v1.csv"),
    )
    parser.add_argument(
        "--features",
        type=str,
        default=str(Path("data") / "processed" / "deepesense_features_v1.csv"),
    )
    parser.add_argument(
        "--gnn-ckpt",
        type=str,
        default=str(Path("models") / "gnn_bandgap_v1.pt"),
    )
    parser.add_argument(
        "--xgb-model",
        type=str,
        default=str(Path("models") / "baseline_xgboost_v1.json"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("results") / "compare_gnn_xgboost_top10_v1.csv"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()
    _configure_logging(args.verbose)

    device = torch.device(args.device)
    disc = pd.read_csv(args.discoveries)
    if disc.empty:
        raise ValueError("Discoveries CSV is empty.")
    disc = disc.sort_values("discovery_score", ascending=False).head(int(args.top_k))
    ids: Set[str] = set(disc["material_id"].astype(str))

    model_gnn, cfg = load_gnn_checkpoint(Path(args.gnn_ckpt), device)
    cutoff = float(cfg["cutoff"])
    num_gaussians = int(cfg["num_gaussians"])
    max_neighbors = int(cfg.get("max_num_neighbors", 32))

    structs = load_structures_for_ids(Path(args.master), ids)

    df_feat = pd.read_csv(args.features)
    bg_col = _select_bandgap_column(df_feat)
    feat_cols = _select_feature_columns(df_feat, bg_col)
    df_sub = df_feat[df_feat["material_id"].astype(str).isin(ids)].copy()

    xgb = XGBRegressor()
    xgb.load_model(str(Path(args.xgb_model)))

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
            med = df_sub[feat_cols].median(numeric_only=True)
            x = mrow[feat_cols].fillna(med).iloc[0:1]
            xgb_pred = float(xgb.predict(x)[0])

        rows.append(
            {
                "material_id": mid,
                "formula": r.get("formula", ""),
                "utility_tier": r.get("utility_tier", ""),
                "discovery_score": r.get("discovery_score", float("nan")),
                "latent_xgb_eV": r.get("predicted_latent_bandgap_eV", float("nan")),
                "refit_xgb_tabular_eV": xgb_pred,
                "gnn_predicted_eV": gnn_pred,
                "delta_gnn_minus_xgb": (gnn_pred - xgb_pred) if gnn_pred == gnn_pred and xgb_pred == xgb_pred else float("nan"),
            }
        )

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    LOG.warning("Wrote comparison table to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
