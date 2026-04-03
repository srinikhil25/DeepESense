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
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
from tqdm import tqdm

LOG = logging.getLogger("deepesense.train_gnn")

MAX_Z = 118


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


def _pauling_en(symbol: str) -> float:
    """Pauling EN; noble gases often lack X in pymatgen — use 0.0 (avoids NaN + UserWarning spam)."""
    try:
        x = Element(symbol).X
        if x is None:
            return 0.0
        xf = float(x)
        if math.isnan(xf):
            return 0.0
        return xf
    except Exception:
        return 0.0


def _cap_neighbors_per_node(
    edge_index: torch.Tensor,
    dist: torch.Tensor,
    num_nodes: int,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep up to max_neighbors shortest edges per source node."""
    rows = edge_index[0].tolist()
    cols = edge_index[1].tolist()
    ds = dist.tolist()
    buckets: List[List[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for r, c, dd in zip(rows, cols, ds):
        if r < 0 or r >= num_nodes:
            continue
        buckets[r].append((c, dd))
    new_src: List[int] = []
    new_dst: List[int] = []
    new_d: List[float] = []
    for r, pairs in enumerate(buckets):
        pairs.sort(key=lambda t: t[1])
        for c, dd in pairs[:max_neighbors]:
            new_src.append(r)
            new_dst.append(c)
            new_d.append(dd)
    if not new_src:
        return edge_index, dist
    ei = torch.tensor([new_src, new_dst], dtype=torch.long)
    d_out = torch.tensor(new_d, dtype=torch.float)
    return ei, d_out


def structure_to_data(
    struct: Structure,
    cutoff: float,
    bandgap_eV: float,
    num_gaussians: int,
    max_num_neighbors: int = 32,
) -> Optional[Data]:
    """Crystal graph: nodes Z + EN; edges from pymatgen neighbor list (no torch-cluster)."""
    try:
        if len(struct) == 0:
            return None
        pos = torch.tensor(np.asarray(struct.cart_coords, dtype=np.float64), dtype=torch.float)
        n = pos.size(0)
        zs = []
        ens = []
        for site in struct:
            el = site.specie.symbol
            zs.append(Element(el).Z)
            ens.append(_pauling_en(el))
        z_tensor = torch.tensor(zs, dtype=torch.long).clamp(1, MAX_Z)
        en_tensor = torch.tensor(ens, dtype=torch.float).view(-1, 1)

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
        return Data(
            pos=pos,
            z=z_tensor,
            x_en=en_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )
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
    ):
        super().__init__()
        self.atom_embed = nn.Embedding(num_species + 1, embed_dim, padding_idx=0)
        self.lin0 = nn.Linear(embed_dim + 1, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(CGConv(hidden, dim=edge_dim, batch_norm=True))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.lin1 = nn.Linear(hidden, hidden // 2)
        self.lin2 = nn.Linear(hidden // 2, 1)

    def forward(self, data: Data) -> torch.Tensor:
        z = data.z
        en = data.x_en
        ae = self.atom_embed(z.clamp(0, MAX_Z))
        x = torch.cat([ae, en], dim=-1)
        x = self.lin0(x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
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
    """Stream-read master CSV; build graphs only for bandgap > 0."""
    graphs: List[Data] = []
    head = pd.read_csv(master_csv, nrows=1)
    bg_name = _select_bandgap_column(head)
    cols = ["material_id", "structure", bg_name]
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
            )
            if g is not None:
                graphs.append(g)
    return graphs


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

    idx = list(range(len(graphs)))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42)
    train_list = [graphs[i] for i in tr_idx]
    test_list = [graphs[i] for i in te_idx]
    LOG.info("Train graphs: %d | Test graphs: %d", len(train_list), len(test_list))

    loader_kw = dict(
        num_workers=max(0, int(loader_workers)),
        persistent_workers=loader_workers > 0,
    )
    if loader_workers <= 0:
        loader_kw.pop("persistent_workers", None)
    train_loader = DataLoader(
        train_list, batch_size=batch_size, shuffle=True, **loader_kw
    )
    test_loader = DataLoader(
        test_list, batch_size=batch_size, shuffle=False, **loader_kw
    )

    edge_dim = num_gaussians
    model = CrystalGNN(
        edge_dim=edge_dim,
        hidden=hidden,
        num_layers=num_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    accum_steps = max(1, int(accum_steps))
    best_mae = float("inf")
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
                opt.step()
                opt.zero_grad()
        avg_loss = total_loss / max(n_batches, 1)

        model.eval()
        all_pred, all_y = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                p = model(batch)
                all_pred.append(p.cpu())
                all_y.append(batch.y.cpu())
        y_p = torch.cat(all_pred).numpy()
        y_t = torch.cat(all_y).numpy()
        mae = mean_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)
        LOG.info(
            "Epoch %d | train_loss=%.5f | test_MAE=%.5f | test_R2=%.4f",
            epoch + 1,
            avg_loss,
            mae,
            r2,
        )
        if empty_gpu_cache_each_epoch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if mae < best_mae:
            best_mae = mae
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
                    },
                },
                model_out,
            )
            LOG.info("Saved best checkpoint (MAE=%.5f) to %s", mae, model_out)

    LOG.warning("Training finished. Best test MAE: %.5f", best_mae)


def load_gnn_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[CrystalGNN, dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = CrystalGNN(
        edge_dim=int(cfg["edge_dim"]),
        hidden=int(cfg["hidden"]),
        num_layers=int(cfg["num_layers"]),
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
    loader = DataLoader([g], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        return float(model(batch).item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Train crystal GNN (CGCNN-style) for bandgap prediction.")
    parser.add_argument(
        "--master",
        type=str,
        default=str(Path("data") / "raw" / "deepesense_master_v1.csv"),
        help="Master CSV with structure JSON and bandgap.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("models") / "gnn_bandgap_v1.pt"),
        help="Output checkpoint (.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (adjust for time vs convergence).",
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
