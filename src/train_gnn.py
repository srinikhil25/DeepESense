from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pymatgen.core import Element, Structure
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
from tqdm import tqdm

from .holdout_split import (
    DEFAULT_SPLIT_PATH,
    TEST,
    TRAIN,
    VAL,
    bucket_series,
    load_or_create_split,
)

LOG = logging.getLogger("deepesense.train_gnn")

MAX_Z = 118
NUM_SCALAR_FEATS = 4  # [Pauling EN, period/7, group/18, covalent_radius/3.0]


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _select_bandgap_column(df: pd.DataFrame) -> str:
    if "bandgap_eV" in df.columns:
        return "bandgap_eV"
    if "band_gap" in df.columns:
        return "band_gap"
    raise ValueError("No bandgap column found. Expected 'bandgap_eV' or 'band_gap'.")


def gaussian_distance_expansion(
    distances: torch.Tensor, num_gaussians: int = 16, cutoff: float = 5.0
) -> torch.Tensor:
    """RBF on pairwise distances (SchNet / CGCNN-style edge features)."""
    device = distances.device
    centers = torch.linspace(0.0, cutoff, num_gaussians, device=device)
    width = (cutoff / num_gaussians) ** 2 * 2.0
    diff = distances.unsqueeze(-1) - centers.unsqueeze(0)
    return torch.exp(-(diff ** 2) / width)


def _safe_float(x: object, fallback: float = 0.0) -> float:
    try:
        if x is None:
            return fallback
        xf = float(x)
        if math.isnan(xf):
            return fallback
        return xf
    except Exception:
        return fallback


def _atom_scalar_features(symbol: str) -> List[float]:
    """
    Physics-aware per-atom scalar features:
      [Pauling EN, period/7, group/18, covalent_radius/3.0]

    All normalized to roughly [0, 1]. Noble gases / rare cases with missing
    values fall back to 0.0 (safer than NaN through the GNN).
    """
    try:
        el = Element(symbol)
        en = _safe_float(el.X, 0.0)
        period = _safe_float(getattr(el, "row", None), 0.0) / 7.0
        group = _safe_float(getattr(el, "group", None), 0.0) / 18.0
        cov_r = _safe_float(getattr(el, "atomic_radius", None), 1.0) / 3.0
        return [en, period, group, cov_r]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


def _cap_neighbors_per_node(
    edge_index: torch.Tensor,
    dist: torch.Tensor,
    num_nodes: int,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Keep up to max_neighbors shortest edges per source node, then **symmetrize**
    by taking the union of forward + reverse edges. CGConv assumes a symmetric
    message-passing graph — a one-sided cap can silently break that invariant.
    """
    rows = edge_index[0].tolist()
    cols = edge_index[1].tolist()
    ds = dist.tolist()
    buckets: List[List[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for r, c, dd in zip(rows, cols, ds):
        if r < 0 or r >= num_nodes:
            continue
        buckets[r].append((c, dd))

    # Step 1: per-source cap keyed on distance.
    kept: dict[tuple[int, int], float] = {}
    for r, pairs in enumerate(buckets):
        pairs.sort(key=lambda t: t[1])
        for c, dd in pairs[:max_neighbors]:
            key = (r, c)
            prev = kept.get(key)
            if prev is None or dd < prev:
                kept[key] = dd

    if not kept:
        return edge_index, dist

    # Step 2: symmetrize. For every kept (i, j), ensure (j, i) also exists.
    # Use the original min distance so reverse edges carry a sensible edge_attr.
    reverse_add: dict[tuple[int, int], float] = {}
    for (i, j), dd in kept.items():
        rev = (j, i)
        if rev not in kept and rev not in reverse_add:
            reverse_add[rev] = dd
    kept.update(reverse_add)

    new_src: List[int] = []
    new_dst: List[int] = []
    new_d: List[float] = []
    for (r, c), dd in kept.items():
        new_src.append(r)
        new_dst.append(c)
        new_d.append(dd)

    ei = torch.tensor([new_src, new_dst], dtype=torch.long)
    d_out = torch.tensor(new_d, dtype=torch.float)
    return ei, d_out


def structure_to_data(
    struct: Structure,
    cutoff: float,
    bandgap_eV: float,
    num_gaussians: int,
    max_num_neighbors: int = 32,
    material_id: Optional[str] = None,
) -> Optional[Data]:
    """Crystal graph: nodes Z + physics-scalar features; edges from pymatgen neighbor list."""
    try:
        if len(struct) == 0:
            return None
        pos = torch.tensor(np.asarray(struct.cart_coords, dtype=np.float64), dtype=torch.float)
        n = pos.size(0)
        zs: List[int] = []
        scalar_rows: List[List[float]] = []
        for site in struct:
            el = site.specie.symbol
            zs.append(Element(el).Z)
            scalar_rows.append(_atom_scalar_features(el))
        z_tensor = torch.tensor(zs, dtype=torch.long).clamp(1, MAX_Z)
        x_scalar = torch.tensor(scalar_rows, dtype=torch.float).view(-1, NUM_SCALAR_FEATS)

        c_idx, n_idx, _, dists = struct.get_neighbor_list(cutoff)
        if len(c_idx) == 0 and n > 0:
            edge_index = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
            d = torch.zeros(n, dtype=torch.float)
        else:
            edge_index = torch.tensor(np.stack([c_idx, n_idx]), dtype=torch.long)
            d = torch.tensor(np.asarray(dists, dtype=np.float64), dtype=torch.float)
            if max_num_neighbors > 0 and edge_index.size(1) > 0:
                edge_index, d = _cap_neighbors_per_node(edge_index, d, n, max_num_neighbors)

        d = d.clamp(min=1e-8)
        edge_attr = gaussian_distance_expansion(d, num_gaussians=num_gaussians, cutoff=cutoff)

        y = torch.tensor([bandgap_eV], dtype=torch.float)
        data = Data(
            pos=pos,
            z=z_tensor,
            x_scalar=x_scalar,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )
        if material_id is not None:
            data.material_id = str(material_id)
        return data
    except Exception as e:
        LOG.debug("structure_to_data failed: %s", e)
        return None


def schneider_bandgap_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_under: float = 2.0,
    high_bg_thresh: float = 4.0,
) -> torch.Tensor:
    """
    Schneider-style asymmetric loss: extra penalty when underestimating high bandgaps
    (e.g. wide-gap ionic insulators like LiF).
    """
    err = pred - target
    mae = err.abs()
    high = (target > high_bg_thresh).float()
    under = (pred < target).float()
    extra = high * under * (target - pred).clamp(min=0.0) ** 2
    return mae.mean() + lambda_under * extra.mean()


class CrystalGNN(nn.Module):
    """CGCNN-style crystal graph network with CGConv + distance RBF edge attributes."""

    def __init__(
        self,
        num_species: int = MAX_Z + 1,
        embed_dim: int = 64,
        edge_dim: int = 16,
        hidden: int = 128,
        num_layers: int = 3,
        num_scalar_feats: int = NUM_SCALAR_FEATS,
    ):
        super().__init__()
        self.num_scalar_feats = int(num_scalar_feats)
        self.atom_embed = nn.Embedding(num_species + 1, embed_dim, padding_idx=0)
        self.lin0 = nn.Linear(embed_dim + self.num_scalar_feats, hidden)
        # IMPORTANT: CGConv's built-in batch_norm=True was the root cause of
        # the "train loss descends smoothly but val_R² crashes to -thousands
        # on a handful of samples" pathology we saw on our first training run.
        # With physical batch_size=16 and widely varying graph sizes, a single
        # pathological batch can pollute BN's running_var with a near-zero
        # value on some hidden feature. In eval mode, dividing by sqrt(~0+eps)
        # produces huge outputs for any val sample that excites that feature,
        # while train-mode forward (which uses current-batch statistics) stays
        # healthy. We replace it with LayerNorm, which normalizes per-sample
        # across the feature dim — no running stats, no train/eval switch, no
        # dependence on batch composition. Standard modern GNN practice.
        self.convs = nn.ModuleList(
            [CGConv(hidden, dim=edge_dim, batch_norm=False) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(num_layers)]
        )
        self.lin1 = nn.Linear(hidden, hidden // 2)
        self.lin2 = nn.Linear(hidden // 2, 1)

    def forward(self, data: Data) -> torch.Tensor:
        z = data.z
        x_scalar = data.x_scalar
        # Back-compat for older Data objects that used x_en (1-dim EN only).
        if x_scalar is None and hasattr(data, "x_en"):
            x_scalar = data.x_en
        ae = self.atom_embed(z.clamp(0, MAX_Z))
        x = torch.cat([ae, x_scalar], dim=-1)
        x = self.lin0(x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)

        x = global_mean_pool(x, data.batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(-1)


def load_labeled_graphs(
    master_csv: Path,
    cutoff: float,
    num_gaussians: int,
    max_num_neighbors: int,
    max_samples: Optional[int] = None,
    chunksize: int = 2000,
) -> List[Data]:
    """Stream-read master CSV; build graphs only for bandgap > 0. Carries material_id on each graph."""
    graphs: List[Data] = []
    head = pd.read_csv(master_csv, nrows=1)
    bg_name = _select_bandgap_column(head)
    has_formula = "formula" in head.columns
    cols = ["material_id", "structure", bg_name]
    if has_formula:
        cols.append("formula")
    reader = pd.read_csv(master_csv, usecols=cols, chunksize=chunksize)
    for chunk in reader:
        chunk[bg_name] = pd.to_numeric(chunk[bg_name], errors="coerce")
        sub = chunk[chunk[bg_name] > 0]
        for _, row in sub.iterrows():
            if max_samples is not None and len(graphs) >= max_samples:
                return graphs
            raw = row["structure"]
            if pd.isna(raw) or str(raw).strip() == "":
                continue
            try:
                d = json.loads(raw) if isinstance(raw, str) else raw
                struct = Structure.from_dict(d)
            except Exception:
                continue
            bg = float(row[bg_name])
            g = structure_to_data(
                struct,
                cutoff=cutoff,
                bandgap_eV=bg,
                num_gaussians=num_gaussians,
                max_num_neighbors=max_num_neighbors,
                material_id=str(row["material_id"]),
            )
            if g is None:
                continue
            if has_formula:
                g.formula = str(row["formula"]) if pd.notna(row["formula"]) else ""
            graphs.append(g)
    return graphs


def _split_graphs_by_chemsys(
    graphs: List[Data],
    split: dict,
) -> tuple[List[Data], List[Data], List[Data]]:
    """Partition graphs into train/val/test using the persisted chemsys map."""
    if not graphs:
        return [], [], []
    formulas = [getattr(g, "formula", "") for g in graphs]
    mids = [getattr(g, "material_id", "") for g in graphs]
    df = pd.DataFrame({"material_id": mids, "formula": formulas})
    buckets = bucket_series(df, split).values
    tr, va, te = [], [], []
    for g, b in zip(graphs, buckets):
        if b == TRAIN:
            tr.append(g)
        elif b == VAL:
            va.append(g)
        elif b == TEST:
            te.append(g)
        else:
            tr.append(g)
    return tr, va, te


def _eval_loader(
    model: CrystalGNN, loader: DataLoader, device: torch.device
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_pred, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            p = model(batch)
            all_pred.append(p.cpu())
            all_y.append(batch.y.cpu())
    if not all_pred:
        return float("nan"), float("nan"), np.array([]), np.array([])
    y_p = torch.cat(all_pred).numpy()
    y_t = torch.cat(all_y).numpy()
    return float(mean_absolute_error(y_t, y_p)), float(r2_score(y_t, y_p)), y_p, y_t


def train_gnn(
    master_csv: Path,
    model_out: Path,
    epochs: int,
    batch_size: int,
    accum_steps: int,
    lr: float,
    cutoff: float,
    num_gaussians: int,
    max_num_neighbors: int,
    num_layers: int,
    hidden: int,
    lambda_under: float,
    high_bg_thresh: float,
    max_samples: Optional[int],
    loader_workers: int,
    empty_gpu_cache_each_epoch: bool,
    device: torch.device,
    split_path: Path,
    patience: int,
) -> None:
    LOG.info("Loading labeled crystal graphs from %s", master_csv)
    graphs = load_labeled_graphs(
        master_csv,
        cutoff=cutoff,
        num_gaussians=num_gaussians,
        max_num_neighbors=max_num_neighbors,
        max_samples=max_samples,
    )
    if len(graphs) < 100:
        raise ValueError(f"Too few graphs after filtering: {len(graphs)}")

    # Load or extend the shared chemsys holdout split.
    df_for_split = pd.DataFrame(
        {
            "material_id": [getattr(g, "material_id", "") for g in graphs],
            "formula": [getattr(g, "formula", "") for g in graphs],
        }
    )
    if not (df_for_split["formula"].astype(str).str.len() > 0).any():
        raise ValueError(
            "Master CSV must carry a 'formula' column for the chemsys holdout split."
        )
    split = load_or_create_split(df_for_split, path=split_path)
    train_list, val_list, test_list = _split_graphs_by_chemsys(graphs, split)
    LOG.warning(
        "GNN split (graphs): train=%d | val=%d | test=%d",
        len(train_list), len(val_list), len(test_list),
    )
    if len(train_list) == 0 or len(val_list) == 0:
        raise ValueError(
            "Need non-empty train and val sets after chemsys split. "
            "Got train=%d, val=%d." % (len(train_list), len(val_list))
        )

    loader_kw = dict(
        num_workers=max(0, int(loader_workers)),
        persistent_workers=loader_workers > 0,
    )
    if loader_workers <= 0:
        loader_kw.pop("persistent_workers", None)
    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False, **loader_kw)
    test_loader = (
        DataLoader(test_list, batch_size=batch_size, shuffle=False, **loader_kw)
        if test_list
        else None
    )

    edge_dim = num_gaussians
    model = CrystalGNN(
        edge_dim=edge_dim,
        hidden=hidden,
        num_layers=num_layers,
        num_scalar_feats=NUM_SCALAR_FEATS,
    ).to(device)
    # AdamW (decoupled weight decay) + gradient clipping are both mandatory
    # here because the Schneider loss has a quadratic underestimation term
    # on wide-gap samples: a single LiF-like target (bg≈14 eV) predicted as
    # ~0 early in training yields a gradient magnitude of order 2·λ·err on
    # the readout layer, which plain Adam cannot damp because its second-
    # moment estimate is still bias-corrected toward zero. Without clipping,
    # that single sample shoves the model into a different region of
    # parameter space and we see val_MAE oscillate between 0.9 and 7.0 eV
    # epoch-to-epoch even while the per-batch *mean* train loss looks smooth.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2, min_lr=1e-5,
    )
    grad_clip_max_norm = 1.0

    accum_steps = max(1, int(accum_steps))
    best_val_mae = float("inf")
    best_state: Optional[dict] = None
    best_epoch = -1
    patience_left = int(patience)
    n_train_batches = len(train_loader)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        opt.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            pred = model(batch)
            loss = schneider_bandgap_loss(
                pred, batch.y, lambda_under=lambda_under, high_bg_thresh=high_bg_thresh
            )
            (loss / accum_steps).backward()
            total_loss += loss.item()
            n_batches += 1
            is_update_step = (step + 1) % accum_steps == 0 or (step + 1) == n_train_batches
            if is_update_step:
                # Clip the accumulated gradient *before* the step. max_norm=1.0
                # is a standard, conservative choice for CGCNN-scale nets and
                # is what stops the Schneider quadratic term from blowing up
                # the readout layer on a single wide-gap sample.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                opt.step()
                opt.zero_grad()
        avg_loss = total_loss / max(n_batches, 1)

        val_mae, val_r2, _, _ = _eval_loader(model, val_loader, device)
        current_lr = opt.param_groups[0]["lr"]
        LOG.warning(
            "Epoch %d | train_loss=%.5f | val_MAE=%.5f | val_R2=%.4f | lr=%.2e",
            epoch + 1, avg_loss, val_mae, val_r2, current_lr,
        )
        # Halve LR when val MAE stalls for 2 epochs. Combined with clipping
        # this turns oscillation into monotone descent in practice.
        scheduler.step(val_mae)

        if empty_gpu_cache_each_epoch and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(patience)
            LOG.info("  ↑ new best val MAE = %.5f (epoch %d)", val_mae, best_epoch)
        else:
            patience_left -= 1
            LOG.info("  patience left: %d", patience_left)
            if patience_left <= 0:
                LOG.warning("Early stopping at epoch %d (best val MAE=%.5f @ epoch %d)",
                            epoch + 1, best_val_mae, best_epoch)
                break

    # Reload best-by-val weights and compute final TEST metrics (never used
    # for early stopping or model selection).
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    if test_loader is not None:
        test_mae, test_r2, _, _ = _eval_loader(model, test_loader, device)
    else:
        test_mae, test_r2 = float("nan"), float("nan")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "cutoff": cutoff,
                "num_gaussians": num_gaussians,
                "max_num_neighbors": max_num_neighbors,
                "num_layers": num_layers,
                "hidden": hidden,
                "edge_dim": edge_dim,
                "max_z": MAX_Z,
                "num_scalar_feats": NUM_SCALAR_FEATS,
            },
            "metrics": {
                "best_epoch": best_epoch,
                "best_val_mae": best_val_mae,
                "test_mae": test_mae,
                "test_r2": test_r2,
                "n_train": len(train_list),
                "n_val": len(val_list),
                "n_test": len(test_list),
                "split_method": split.get("method", "by_chemsys_hash"),
                "split_seed": int(split.get("seed", 42)),
                "split_path": str(split_path),
            },
        },
        model_out,
    )
    # Write a .meta.json sidecar (same convention as train_baseline) so
    # ``scripts/log_run.py --ingest`` can pull GNN metrics into the run
    # ledger for the paper-ready headline numbers block.
    meta = {
        "best_epoch": best_epoch,
        "best_val_mae": round(best_val_mae, 5),
        "test_mae": round(test_mae, 5) if test_mae == test_mae else None,
        "test_r2": round(test_r2, 4) if test_r2 == test_r2 else None,
        "n_train": len(train_list),
        "n_val": len(val_list),
        "n_test": len(test_list),
        "split_method": split.get("method", "by_chemsys_hash"),
        "split_seed": int(split.get("seed", 42)),
        "split_path": str(split_path),
        "cutoff": cutoff,
        "num_gaussians": num_gaussians,
        "num_layers": num_layers,
        "hidden": hidden,
        "lr": lr,
        "lambda_under": lambda_under,
    }
    meta_path = model_out.with_suffix(model_out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    LOG.warning(
        "Training finished. best_epoch=%d | best_val_MAE=%.5f | test_MAE=%.5f | test_R2=%.4f",
        best_epoch, best_val_mae, test_mae, test_r2,
    )
    LOG.warning("Checkpoint saved to %s", model_out.as_posix())
    LOG.warning("Meta sidecar written to %s", meta_path.as_posix())


def load_gnn_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[CrystalGNN, dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    # Back-compat: old checkpoints only used 1-dim EN scalar.
    num_scalar_feats = int(cfg.get("num_scalar_feats", 1))
    model = CrystalGNN(
        edge_dim=int(cfg["edge_dim"]),
        hidden=int(cfg["hidden"]),
        num_layers=int(cfg["num_layers"]),
        num_scalar_feats=num_scalar_feats,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, cfg


def predict_bandgap_gnn(
    struct: Structure,
    model: CrystalGNN,
    device: torch.device,
    cutoff: float,
    num_gaussians: int,
    max_num_neighbors: int = 32,
) -> float:
    g = structure_to_data(
        struct,
        cutoff=cutoff,
        bandgap_eV=0.0,
        num_gaussians=num_gaussians,
        max_num_neighbors=max_num_neighbors,
    )
    if g is None:
        return float("nan")
    # If the model was trained with only 1 scalar feat (legacy), collapse
    # x_scalar to its first column so the linear layer shapes match.
    if getattr(model, "num_scalar_feats", NUM_SCALAR_FEATS) != g.x_scalar.size(-1):
        g.x_scalar = g.x_scalar[:, : model.num_scalar_feats]
    loader = DataLoader([g], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        return float(model(batch).item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Train crystal GNN (CGCNN-style) for bandgap prediction.")
    parser.add_argument(
        "--master",
        type=str,
        default=str(Path("data") / "raw" / "deepesense_candidates.csv"),
        help="Master CSV with structure JSON and bandgap.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("models") / "cgcnn.pt"),
        help="Output checkpoint (.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs (early stopping may end sooner).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early-stopping patience on val MAE (epochs without improvement).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Graphs per GPU step. Lower if you hit CUDA OOM (try 4 or 2 on 8GB VRAM).",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch ≈ batch-size × accum-steps). Use 2–4 to mimic larger batches without extra VRAM.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cutoff", type=float, default=5.0, help="Neighbor graph cutoff (Å).")
    parser.add_argument("--num-gaussians", type=int, default=16)
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=32,
        help="Caps edges per node in radius graph; lower (16–24) saves GPU memory on large cells.",
    )
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="Hidden width; try 96 or 64 if still OOM after lowering batch-size.",
    )
    parser.add_argument("--lambda-under", type=float, default=2.0, help="Schneider-style high-gap underestimation penalty.")
    parser.add_argument("--high-bg-thresh", type=float, default=4.0, help="Bandgap (eV) above which underestimation is penalized more.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap labeled graphs for debugging.")
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is safest on Windows + limited RAM).",
    )
    parser.add_argument(
        "--empty-gpu-cache",
        action="store_true",
        help="Call torch.cuda.empty_cache() after each epoch to reduce peak VRAM fragmentation.",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default=str(DEFAULT_SPLIT_PATH),
        help="Persisted chemsys holdout split JSON (created if missing).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    _configure_logging(args.verbose)
    device = torch.device(args.device)

    train_gnn(
        master_csv=Path(args.master),
        model_out=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        cutoff=args.cutoff,
        num_gaussians=args.num_gaussians,
        max_num_neighbors=args.max_neighbors,
        num_layers=args.num_layers,
        hidden=args.hidden,
        lambda_under=args.lambda_under,
        high_bg_thresh=args.high_bg_thresh,
        max_samples=args.max_samples,
        loader_workers=args.loader_workers,
        empty_gpu_cache_each_epoch=args.empty_gpu_cache,
        device=device,
        split_path=Path(args.split_path),
        patience=args.patience,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
