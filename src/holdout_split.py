"""
DeepESense held-out chemistry split.

Random row splits leak information between train and test because multiple MP
entries often share the same chemical system (e.g. several polymorphs of
Fe2O3). A model that memorizes chemistry-level priors can then look better
than it really is on held-out rows.

This module builds a **by-chemical-system** split instead: every distinct
chemical system (frozenset of element symbols, canonicalized as a sorted
"Fe-O" string) is deterministically hashed into one of ``train`` / ``val`` /
``test``. All rows whose chemistry falls in the same bucket travel together.

Key properties:

- Deterministic (seed + chemsys hash), so every script in the pipeline
  resolves the same bucket for the same material without coordinating.
- Persisted to disk as ``models/holdout_split.json``. Any script may
  extend the map with newly-seen chemsys (again via the seeded hash), so
  Stage-1 / Stage-2 / GNN / tabular all agree even if they ingest slightly
  different row sets.
- Reportable. The JSON records method, seed, fractions, per-chemsys buckets,
  and per-bucket counts — enough for a reviewer to reproduce the exact split.

Paper rationale: the headline Stage-1 / Stage-2 / GNN metrics reported in
``reports/`` must come from the **test** bucket only; train/val are used for
fitting and early stopping respectively. The QE validation bridge is a
separate (and stronger) oracle on top of that test bucket.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from pymatgen.core import Composition

LOG = logging.getLogger("deepesense.holdout_split")

TRAIN = "train"
VAL = "val"
TEST = "test"
BUCKETS = (TRAIN, VAL, TEST)

DEFAULT_SPLIT_PATH = Path("models") / "holdout_split.json"
DEFAULT_SEED = 42
DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15


def chemsys_from_formula(formula: object) -> Optional[str]:
    """Canonical chemical system key: sorted element symbols joined by '-'."""
    if formula is None:
        return None
    try:
        comp = Composition(str(formula))
        els = sorted({str(e.symbol) for e in comp.elements})
        if not els:
            return None
        return "-".join(els)
    except Exception:
        return None


def _hash_unit(key: str, seed: int) -> float:
    """Stable [0, 1) hash of (seed, key) via SHA-256 — no numpy RNG, no drift."""
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def assign_bucket(chemsys: str, seed: int, val_frac: float, test_frac: float) -> str:
    u = _hash_unit(chemsys, seed)
    if u < test_frac:
        return TEST
    if u < test_frac + val_frac:
        return VAL
    return TRAIN


def _formula_col(df: pd.DataFrame) -> str:
    for c in ("formula", "pretty_formula", "formula_pretty", "reduced_formula"):
        if c in df.columns:
            return c
    raise ValueError(
        "Holdout split needs a formula column in the input DataFrame "
        "(looked for: formula, pretty_formula, formula_pretty, reduced_formula)."
    )


def load_or_create_split(
    df: pd.DataFrame,
    path: Path = DEFAULT_SPLIT_PATH,
    seed: int = DEFAULT_SEED,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> dict:
    """
    Load the persisted split, or build it from ``df`` if missing. Always
    extends the chemsys→bucket map with any chemsys present in ``df`` but
    not yet persisted, and rewrites the JSON if anything changed.
    """
    path = Path(path)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        # If someone accidentally edits the fractions on disk, keep them.
        data.setdefault("seed", int(seed))
        data.setdefault("val_frac", float(val_frac))
        data.setdefault("test_frac", float(test_frac))
        data.setdefault("bucket_by_chemsys", {})
        created = False
    else:
        data = {
            "method": "by_chemsys_hash",
            "seed": int(seed),
            "val_frac": float(val_frac),
            "test_frac": float(test_frac),
            "bucket_by_chemsys": {},
        }
        created = True

    bucket_map: Dict[str, str] = data["bucket_by_chemsys"]
    col = _formula_col(df)
    added = 0
    for f in df[col].dropna().astype(str).unique():
        cs = chemsys_from_formula(f)
        if cs is None or cs in bucket_map:
            continue
        bucket_map[cs] = assign_bucket(
            cs, int(data["seed"]), float(data["val_frac"]), float(data["test_frac"])
        )
        added += 1

    # Recount.
    counts = {b: 0 for b in BUCKETS}
    for b in bucket_map.values():
        counts[b] = counts.get(b, 0) + 1
    data["bucket_by_chemsys"] = bucket_map
    data["n_chemsys"] = len(bucket_map)
    data["n_chemsys_by_bucket"] = counts

    if created or added:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        LOG.warning(
            "Holdout split at %s: %d chemsys total (%d new). train/val/test = %d/%d/%d",
            path.as_posix(),
            len(bucket_map),
            added,
            counts[TRAIN],
            counts[VAL],
            counts[TEST],
        )
    else:
        LOG.info(
            "Holdout split loaded from %s: %d chemsys, train/val/test = %d/%d/%d",
            path.as_posix(),
            len(bucket_map),
            counts[TRAIN],
            counts[VAL],
            counts[TEST],
        )
    return data


def bucket_series(df: pd.DataFrame, split: dict) -> pd.Series:
    """
    Return a pd.Series[str] aligned to df.index giving the bucket for each
    row. Rows with unknown chemsys fall back to TRAIN (safe default —
    they'll contribute to training but never to held-out metrics).
    """
    col = _formula_col(df)
    bucket_map: Dict[str, str] = split["bucket_by_chemsys"]
    seed = int(split["seed"])
    val_frac = float(split["val_frac"])
    test_frac = float(split["test_frac"])

    def _b(f: object) -> str:
        cs = chemsys_from_formula(f)
        if cs is None:
            return TRAIN
        b = bucket_map.get(cs)
        if b is None:
            # Unseen chemsys — derive deterministically so the script never
            # silently drops rows. The caller should normally have extended
            # the persisted map before getting here via load_or_create_split.
            b = assign_bucket(cs, seed, val_frac, test_frac)
        return b

    return df[col].map(_b)
