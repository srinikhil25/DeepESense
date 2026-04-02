from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure import (
    DensityFeatures,
    GlobalSymmetryFeatures,
    RadialDistributionFunction,
)
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from pymatgen.core import Composition, Structure


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


@dataclass(frozen=True)
class FeatureConfig:
    include_structure: bool = True
    include_composition: bool = True


def featurize_candidates(
    candidates_csv: Path,
    out_csv: Path,
    config: FeatureConfig,
    structure_col: str = "structure",
) -> pd.DataFrame:
    """
    Input is expected to be a CSV produced by `src.discovery_engine` containing:
    - `material_id`
    - `formula`
    - `structure` (JSON-serializable dict; pymatgen Structure.as_dict())

    Output is a flat feature table suitable for clustering/novelty detection.
    """
    df = pd.read_csv(candidates_csv)
    if df.empty:
        raise ValueError(f"No rows in {candidates_csv}")

    # ---- Composition (Magpie / ElementProperty preset)
    if config.include_composition:
        LOG.info("Computing Magpie compositional descriptors.")
        df["composition"] = df["formula"].apply(lambda f: Composition(f))
        magpie = ElementProperty.from_preset("magpie")
        df = magpie.featurize_dataframe(df, col_id="composition", ignore_errors=True)

    # ---- Structure descriptors (optional; requires structures present)
    if config.include_structure:
        if structure_col not in df.columns:
            raise ValueError(f"Missing '{structure_col}' column; rerun discovery with --include-structures.")

        LOG.info("Parsing structures from dict form.")
        df["_pmg_structure"] = df[structure_col].apply(
            lambda s: Structure.from_dict(json.loads(s)) if isinstance(s, str) else Structure.from_dict(s)
        )

        LOG.info("Computing structural descriptors.")
        df = DensityFeatures().featurize_dataframe(df, col_id="_pmg_structure", ignore_errors=True)
        df = GlobalSymmetryFeatures().featurize_dataframe(df, col_id="_pmg_structure", ignore_errors=True)

        # Local-environment fingerprint aggregated across sites
        cnn = CrystalNNFingerprint.from_preset("ops")
        ssf = SiteStatsFingerprint(cnn, stats=("mean", "std_dev"))
        df = ssf.featurize_dataframe(df, col_id="_pmg_structure", ignore_errors=True)

        # Coarser structural “shape” descriptor
        rdf = RadialDistributionFunction(cutoff=10.0, bin_size=0.25)
        df = rdf.featurize_dataframe(df, col_id="_pmg_structure", ignore_errors=True)

        df = df.drop(columns=["_pmg_structure"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    LOG.warning("Wrote features to %s", out_csv.as_posix())
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Featurize candidate structures for unsupervised discovery.")
    parser.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=str(Path("data") / "raw" / "mp_candidates_v1.csv"),
        help="Input candidates CSV (from discovery_engine).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data") / "processed" / "mp_candidates_features_v1.csv"),
        help="Output feature CSV path.",
    )
    parser.add_argument(
        "--no-structure",
        action="store_true",
        help="Skip structural descriptors (composition-only).",
    )
    parser.add_argument(
        "--no-composition",
        action="store_true",
        help="Skip compositional descriptors.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG).",
    )
    args = parser.parse_args()
    _configure_logging(args.verbose)

    cfg = FeatureConfig(
        include_structure=not args.no_structure,
        include_composition=not args.no_composition,
    )

    featurize_candidates(
        candidates_csv=Path(args.inp),
        out_csv=Path(args.out),
        config=cfg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
