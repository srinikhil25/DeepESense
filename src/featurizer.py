from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure import DensityFeatures
from pymatgen.core import Composition, Structure
from tqdm.auto import tqdm


LOG = logging.getLogger("deepesense.featurizer")


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


def _set_n_jobs_safe(featurizer: Any, n_jobs: int) -> None:
    set_jobs = getattr(featurizer, "set_n_jobs", None)
    if callable(set_jobs):
        if n_jobs == -1:
            resolved = os.cpu_count() or 1
        else:
            resolved = max(1, int(n_jobs))
        set_jobs(resolved)


def _parse_structure(raw: Any) -> Optional[Structure]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, dict):
        try:
            return Structure.from_dict(raw)
        except Exception:
            return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            return Structure.from_dict(json.loads(s))
        except Exception:
            return None
    return None


def featurize_master(
    inp_csv: Path,
    out_csv: Path,
    n_jobs: int = -1,
    structure_col: str = "structure",
    formula_col: str = "formula",
    id_col: str = "material_id",
) -> pd.DataFrame:
    df = pd.read_csv(inp_csv)
    if df.empty:
        raise ValueError(f"No rows in {inp_csv}")

    for col in (id_col, formula_col, structure_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    tqdm.pandas(desc="Parsing compositions")
    df["composition"] = df[formula_col].progress_apply(lambda f: Composition(f) if pd.notna(f) else None)

    tqdm.pandas(desc="Parsing structures")
    df["_pmg_structure"] = df[structure_col].progress_apply(_parse_structure)

    LOG.info("Computing Magpie composition descriptors (n_jobs=%s).", n_jobs)
    magpie = ElementProperty.from_preset("magpie")
    _set_n_jobs_safe(magpie, n_jobs)
    df = magpie.featurize_dataframe(df, col_id="composition", ignore_errors=True, pbar=True)

    LOG.info("Computing DensityFeatures structural descriptors (n_jobs=%s).", n_jobs)
    density = DensityFeatures()
    _set_n_jobs_safe(density, n_jobs)
    df = density.featurize_dataframe(df, col_id="_pmg_structure", ignore_errors=True, pbar=True)

    if "bandgap_eV" not in df.columns:
        if "band_gap" in df.columns:
            df["bandgap_eV"] = df["band_gap"]
        else:
            df["bandgap_eV"] = pd.NA

    keep_always = [id_col, "bandgap_eV"]
    drop_raw = [c for c in ("structure", "elements", "_pmg_structure", "composition") if c in df.columns]
    df = df.drop(columns=drop_raw)

    # keep material_id and bandgap_eV plus generated features and existing metadata

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    LOG.warning("Wrote features to %s", out_csv.as_posix())
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Featurize DeepESense master dataset.")
    parser.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=str(Path("data") / "raw" / "deepesense_candidates.csv"),
        help="Input master CSV.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data") / "processed" / "deepesense_features.csv"),
        help="Output feature CSV path.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for matminer (-1 = all cores).")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG).",
    )
    args = parser.parse_args()
    _configure_logging(args.verbose)

    featurize_master(
        inp_csv=Path(args.inp),
        out_csv=Path(args.out),
        n_jobs=args.n_jobs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
