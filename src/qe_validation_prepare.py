"""
Validation bridge — Quantum ESPRESSO input preparation (rule-fixed).

Takes a Stage-2 refined CSV (from ``src/compare_gnn_xgboost.py``), selects a
small subset of materials whose ML answers are most informative to validate
with first-principles, pulls each crystal structure from the master corpus,
and writes a self-contained QE input directory under ``dft_validation_qe/``.

Selection priority (in order):
1. ``strong`` GNN/XGB disagreement — at least one model must be wrong, so DFT
   is the cheapest oracle for resolving them.
2. Top ``finite_gap_consensus`` rows by Stage-1 score — these are the most
   confident dark-matter unfoldings; a small DFT confirmation set anchors
   the precision of the pipeline.

The protocol is intentionally fixed and identical for every material so that
DFT results across the validation set are comparable. Parameters live in a
single ``protocol.json`` written at the top of the validation directory; per
material we write:

- ``scf.in``     — self-consistent ground state
- ``nscf.in``    — non-self-consistent on a denser grid for the gap
- ``POSCAR``     — pymatgen reference of the standardized primitive cell
- ``manifest.json`` — provenance for this specific candidate

Pseudopotentials are NOT bundled. Each ``scf.in`` references pseudo files by
name only, and the user must point ``pseudo_dir`` (or set ``ESPRESSO_PSEUDO``)
at a local SSSP / PSlibrary directory before running ``pw.x``. The chosen
filenames default to the SSSP Efficiency v1.3 PBE family; override with
``--pseudo-family`` if needed.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.pwscf import PWInput
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .compare_gnn_xgboost import load_structures_for_ids

LOG = logging.getLogger("deepesense.qe_validation_prepare")


# --- rule-fixed DFT protocol ------------------------------------------------

@dataclass(frozen=True)
class QEProtocol:
    """All parameters that define a reproducible QE validation run.

    These are deliberately conservative and identical across every candidate
    so that gap predictions across the validation set are comparable. They
    are NOT optimized per material — that is the entire point of the
    validation bridge.
    """

    xc_functional: str = "PBE"
    pseudo_family: str = "SSSP_efficiency_v1.3_PBE"
    ecutwfc_Ry: float = 60.0
    ecutrho_Ry: float = 480.0
    kpoint_density_per_inv_ang: float = 0.20  # ~0.2 Å^-1 → ~Γ-centered Monkhorst-Pack
    nscf_kpoint_density_per_inv_ang: float = 0.12  # denser for gap extraction
    occupations: str = "smearing"
    smearing: str = "marzari-vanderbilt"
    degauss_Ry: float = 0.005
    conv_thr: float = 1.0e-8
    mixing_beta: float = 0.4
    electron_maxstep: int = 200
    nbnd_padding: int = 16  # extra empty bands above filled for clean gap

    # --- spin-polarized PBE+U for f-electron systems ------------------------
    # When an f-block element (lanthanide or actinide) is present AND
    # ``enable_plus_u_for_f`` is True, the prepared inputs include:
    #   nspin = 2
    #   starting_magnetization(<f-species idx>) = starting_magnetization_f
    #   lda_plus_u = .true.
    #   lda_plus_u_kind = 0
    #   Hubbard_U(<f-species idx>) = hubbard_u_eV
    # Legacy-syntax Hubbard cards are used because they still parse cleanly in
    # QE 7.x and avoid pymatgen PWInput's lack of HUBBARD-card support. The
    # default U = 6.0 eV is the standard empirical value for 4f localization
    # in lanthanide ionic hosts (cf. Anisimov & Gunnarsson 1991;
    # Larson et al. 2007 for Ln-fluorides).
    enable_plus_u_for_f: bool = False
    hubbard_u_eV: float = 6.0
    starting_magnetization_f: float = 0.5


# f-block (lanthanide + actinide) elements that need Hubbard U under PBE.
# PBE systematically delocalises 4f/5f electrons in ionic hosts and spuriously
# closes the gap — cf. the 4 V3 SCF failures on Cs2PrF6 / RbHoBeF6 / NaPrF4,
# and AFLOW's PBE-0-eV labels on BaTbF6 / Li4TbF8 / YbPO4 / YbBO3 / YbClO
# that trigger our cascade's ``conflict`` verdict.
F_BLOCK_ELEMENTS: Set[str] = {
    # Lanthanides
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    # Actinides (included for completeness; cascade hasn't hit one yet)
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
}


# Minimal SSSP Efficiency v1.3 PBE pseudo filename map. Not exhaustive — the
# script falls back to <Symbol>.UPF for any element not listed here, which the
# user can patch in the protocol manifest before running pw.x.
SSSP_PBE_PSEUDOS: Dict[str, str] = {
    "H": "H_ONCV_PBE-1.0.oncvpsp.upf",
    "Li": "li_pbe_v1.4.uspp.F.UPF",
    "Be": "be_pbe_v1.4.uspp.F.UPF",
    "B": "b_pbe_v1.4.uspp.F.UPF",
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "N": "N.pbe-n-radius_5.UPF",
    "O": "O.pbe-n-kjpaw_psl.0.1.UPF",
    "F": "f_pbe_v1.4.uspp.F.UPF",
    "Na": "na_pbe_v1.5.uspp.F.UPF",
    "Mg": "Mg.pbe-n-kjpaw_psl.0.3.0.UPF",
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
    "P": "P.pbe-n-rrkjus_psl.1.0.0.UPF",
    "S": "s_pbe_v1.4.uspp.F.UPF",
    "Cl": "cl_pbe_v1.4.uspp.F.UPF",
    "K": "K.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ca": "Ca_pbe_v1.uspp.F.UPF",
    "Ti": "ti_pbe_v1.4.uspp.F.UPF",
    "V": "v_pbe_v1.4.uspp.F.UPF",
    "Cr": "cr_pbe_v1.5.uspp.F.UPF",
    "Mn": "mn_pbe_v1.5.uspp.F.UPF",
    "Fe": "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
    "Co": "Co_pbe_v1.2.uspp.F.UPF",
    "Ni": "ni_pbe_v1.4.uspp.F.UPF",
    "Cu": "Cu_pbe_v1.2.uspp.F.UPF",
    "Zn": "Zn_pbe_v1.uspp.F.UPF",
    "Ga": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Ge": "ge_pbe_v1.4.uspp.F.UPF",
    "As": "As.pbe-n-rrkjus_psl.0.2.UPF",
    "Se": "Se_pbe_v1.uspp.F.UPF",
    "Br": "br_pbe_v1.4.uspp.F.UPF",
    "Rb": "Rb_ONCV_PBE-1.0.oncvpsp.upf",
    "Sr": "Sr_pbe_v1.uspp.F.UPF",
    "Y": "Y_pbe_v1.uspp.F.UPF",
    "Zr": "Zr_pbe_v1.uspp.F.UPF",
    "Nb": "Nb.pbe-spn-kjpaw_psl.0.3.0.UPF",
    "Mo": "Mo_ONCV_PBE-1.0.oncvpsp.upf",
    "Ag": "Ag_ONCV_PBE-1.0.oncvpsp.upf",
    "Cd": "Cd.pbe-dn-rrkjus_psl.0.3.1.UPF",
    "In": "In.pbe-dn-rrkjus_psl.0.2.2.UPF",
    "Sn": "Sn_pbe_v1.uspp.F.UPF",
    "Sb": "sb_pbe_v1.4.uspp.F.UPF",
    "Te": "Te_pbe_v1.uspp.F.UPF",
    "I": "I.pbe-n-kjpaw_psl.0.2.UPF",
    "Cs": "Cs_pbe_v1.uspp.F.UPF",
    "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Hf": "Hf-sp.oncvpsp.upf",
    "Ta": "Ta_pbe_v1.uspp.F.UPF",
    "W": "W_pbe_v1.2.uspp.F.UPF",
    "Pt": "pt_pbe_v1.4.uspp.F.UPF",
    "Au": "Au_ONCV_PBE-1.0.oncvpsp.upf",
    "Pb": "Pb.pbe-dn-kjpaw_psl.0.2.2.UPF",
    "Bi": "Bi_pbe_v1.uspp.F.UPF",
    # Lanthanides — Wentzcovitch PAW from SSSP Efficiency v1.3
    "La": "La.paw.z_11.atompaw.wentzcovitch.v1.2.upf",
    "Ce": "Ce.paw.z_12.atompaw.wentzcovitch.v1.2.upf",
    "Pr": "Pr.paw.z_13.atompaw.wentzcovitch.v1.2.upf",
    "Nd": "Nd.paw.z_14.atompaw.wentzcovitch.v1.2.upf",
    "Sm": "Sm.paw.z_16.atompaw.wentzcovitch.v1.2.upf",
    "Eu": "Eu.paw.z_17.atompaw.wentzcovitch.v1.2.upf",
    "Gd": "Gd.paw.z_18.atompaw.wentzcovitch.v1.2.upf",
    "Tb": "Tb.paw.z_19.atompaw.wentzcovitch.v1.2.upf",
    "Dy": "Dy.paw.z_20.atompaw.wentzcovitch.v1.2.upf",
    "Ho": "Ho.paw.z_21.atompaw.wentzcovitch.v1.2.upf",
    "Er": "Er.paw.z_22.atompaw.wentzcovitch.v1.2.upf",
    "Tm": "Tm.paw.z_23.atompaw.wentzcovitch.v1.2.upf",
    "Yb": "Yb.paw.z_24.atompaw.wentzcovitch.v1.2.upf",
    "Lu": "Lu.paw.z_25.atompaw.wentzcovitch.v1.2.upf",
}


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _kpoint_grid_for_lattice(struct: Structure, density_inv_ang: float) -> tuple[int, int, int]:
    """Γ-centered MP grid sized so reciprocal-space sampling is ≤ *density* Å⁻¹.

    The reciprocal lattice vector length along axis *i* is  2π / a_i  (physics
    convention).  The number of k-points needed so that the spacing does not
    exceed *density_inv_ang* is therefore:

        n_i = ceil(2π / (density_inv_ang × a_i))

    We use ``round`` + ``max(1, …)`` for robustness against borderline rounding.
    """
    import math
    TWO_PI = 2.0 * math.pi
    a, b, c = struct.lattice.abc
    return (
        max(1, int(round(TWO_PI / (density_inv_ang * a)))),
        max(1, int(round(TWO_PI / (density_inv_ang * b)))),
        max(1, int(round(TWO_PI / (density_inv_ang * c)))),
    )


def _expected_nbnd(struct: Structure, padding: int) -> int:
    """Rough lower bound for nbnd: 0.6 × valence electrons + padding.

    The 0.6 factor is intentionally generous so we never lose the conduction
    band edge to a too-small nbnd; QE will silently ceiling it for SCF runs.
    """
    nelec = sum(int(site.specie.Z) for site in struct)
    return int(0.6 * nelec) + int(padding)


def _pseudo_map_for_structure(struct: Structure) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for el in {site.specie.symbol for site in struct}:
        out[el] = SSSP_PBE_PSEUDOS.get(el, f"{el}.UPF")
    return out


def _f_block_symbols_in(struct: Structure) -> List[str]:
    """Return f-block symbols present in ``struct`` (sorted, unique)."""
    return sorted({site.specie.symbol for site in struct} & F_BLOCK_ELEMENTS)


def _species_order_from_input(pw_input_text: str) -> List[str]:
    """Parse ATOMIC_SPECIES block to recover the 1-based QE species order."""
    species: List[str] = []
    in_block = False
    for raw in pw_input_text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("ATOMIC_SPECIES"):
            in_block = True
            continue
        if in_block:
            if not stripped:
                continue
            if stripped.startswith(("K_POINTS", "ATOMIC_POSITIONS",
                                    "CELL_PARAMETERS", "OCCUPATIONS",
                                    "CONSTRAINTS", "HUBBARD", "&")):
                break
            parts = stripped.split()
            if parts:
                species.append(parts[0])
    return species


def _inject_plus_u_and_spin(
    pw_in_path: Path,
    f_symbols: List[str],
    hubbard_u_eV: float,
    starting_magnetization: float,
) -> None:
    """Patch a pymatgen-written QE input file with nspin + Hubbard U cards.

    pymatgen's ``PWInput`` has no knob for Hubbard U or spin polarisation, so
    we post-process: discover the species index of each f-block element from
    the already-written ATOMIC_SPECIES block, then splice the extra ``&system``
    keys in just before the namelist's closing ``/``.

    Uses the legacy ``lda_plus_u`` / ``Hubbard_U(i)`` syntax. QE 7.x still
    parses it (emits a deprecation note) and it avoids modelling the
    separate HUBBARD card, which pymatgen can't produce.
    """
    if not f_symbols:
        return

    text = pw_in_path.read_text(encoding="utf-8")
    species_order = _species_order_from_input(text)
    if not species_order:
        LOG.warning("Could not parse ATOMIC_SPECIES from %s; skipping +U injection",
                    pw_in_path)
        return

    inject: List[str] = ["  nspin = 2"]
    for sym in f_symbols:
        if sym not in species_order:
            continue
        idx = species_order.index(sym) + 1  # QE species indices are 1-based
        inject.append(f"  starting_magnetization({idx}) = {starting_magnetization}")

    inject.append("  lda_plus_u = .true.")
    inject.append("  lda_plus_u_kind = 0")
    for sym in f_symbols:
        if sym not in species_order:
            continue
        idx = species_order.index(sym) + 1
        inject.append(f"  Hubbard_U({idx}) = {hubbard_u_eV}")

    # Splice into the &system / &SYSTEM namelist right before its closing "/".
    out_lines: List[str] = []
    in_system = False
    injected = False
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("&system"):
            in_system = True
            out_lines.append(line)
            continue
        if in_system and not injected and stripped == "/":
            out_lines.extend(inject)
            out_lines.append(line)
            injected = True
            in_system = False
            continue
        out_lines.append(line)

    if not injected:
        LOG.warning("Could not find &system terminator in %s; +U not injected",
                    pw_in_path)
        return

    pw_in_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _build_pwinputs(struct: Structure, protocol: QEProtocol) -> tuple[PWInput, PWInput]:
    """Return (scf, nscf) PWInput objects sharing rule-fixed parameters."""
    pseudo = _pseudo_map_for_structure(struct)
    nbnd = _expected_nbnd(struct, protocol.nbnd_padding)

    common_system = {
        "ecutwfc": protocol.ecutwfc_Ry,
        "ecutrho": protocol.ecutrho_Ry,
        "occupations": protocol.occupations,
        "smearing": protocol.smearing,
        "degauss": protocol.degauss_Ry,
        "nbnd": nbnd,
    }
    common_electrons = {
        "conv_thr": protocol.conv_thr,
        "mixing_beta": protocol.mixing_beta,
        "electron_maxstep": protocol.electron_maxstep,
    }

    scf = PWInput(
        structure=struct,
        pseudo=pseudo,
        control={"calculation": "scf", "restart_mode": "from_scratch", "tprnfor": True, "tstress": True},
        system=dict(common_system),
        electrons=dict(common_electrons),
        kpoints_mode="automatic",
        kpoints_grid=_kpoint_grid_for_lattice(struct, protocol.kpoint_density_per_inv_ang),
        kpoints_shift=(0, 0, 0),
    )

    nscf = PWInput(
        structure=struct,
        pseudo=pseudo,
        control={"calculation": "nscf", "restart_mode": "from_scratch", "verbosity": "high"},
        system=dict(common_system),
        electrons=dict(common_electrons),
        kpoints_mode="automatic",
        kpoints_grid=_kpoint_grid_for_lattice(struct, protocol.nscf_kpoint_density_per_inv_ang),
        kpoints_shift=(0, 0, 0),
    )
    return scf, nscf


def _select_rows_by_id(df: pd.DataFrame, ids: List[str]) -> tuple[pd.DataFrame, List[str]]:
    """Pick rows whose ``material_id`` is in ``ids``, preserving the given order."""
    mid_col = df["material_id"].astype(str)
    out_rows: List[pd.Series] = []
    reasons: List[str] = []
    for mid in ids:
        mask = mid_col == str(mid)
        if not mask.any():
            LOG.warning("Requested material_id %s not found in input CSV; skipping", mid)
            continue
        out_rows.append(df[mask].iloc[0])
        reasons.append("explicit_id")
    if not out_rows:
        return df.iloc[0:0].copy(), reasons
    return pd.DataFrame(out_rows).reset_index(drop=True), reasons


def _select_validation_rows(
    df: pd.DataFrame,
    n_disagreement: int,
    n_consensus: int,
    n_random: int = 0,
    random_pool_df: Optional[pd.DataFrame] = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, List[str]]:
    """Pick disagreement rows first, then consensus, then unbiased random samples.

    The random slice is critical for paper-defensible precision numbers:
    disagreement and consensus are biased selection rules, so their precision
    is conditional. Random sampling from the dark-matter pool (or stage-2 set)
    gives an unbiased estimate of the underlying base rate of "actually finite
    gap" materials in PBE-zero-labeled MP entries.
    """
    reasons: List[str] = []
    picks: List[pd.Series] = []
    seen: Set[str] = set()

    if "agreement_class" in df.columns and n_disagreement > 0:
        strong = df[df["agreement_class"] == "strong"].copy()
        if "disagreement_eV" in strong.columns:
            strong = strong.reindex(strong["disagreement_eV"].abs().sort_values(ascending=False).index)
        for _, r in strong.head(n_disagreement).iterrows():
            mid = str(r["material_id"])
            if mid in seen:
                continue
            seen.add(mid)
            picks.append(r)
            reasons.append("strong_disagreement")

    if n_consensus > 0:
        cons = df.copy()
        if "finite_gap_consensus" in cons.columns:
            cons = cons[cons["finite_gap_consensus"]]
        if "stage1_score" in cons.columns:
            cons = cons.sort_values("stage1_score", ascending=False)
        for _, r in cons.iterrows():
            if len([rs for rs in reasons if rs in ("strong_disagreement", "finite_gap_consensus")]) >= n_disagreement + n_consensus:
                break
            mid = str(r["material_id"])
            if mid in seen:
                continue
            seen.add(mid)
            picks.append(r)
            reasons.append("finite_gap_consensus")

    if n_random > 0:
        pool = random_pool_df if random_pool_df is not None else df
        candidates = pool[~pool["material_id"].astype(str).isin(seen)]
        if len(candidates) > 0:
            sampled = candidates.sample(
                n=min(n_random, len(candidates)),
                random_state=random_seed,
            )
            for _, r in sampled.iterrows():
                mid = str(r["material_id"])
                if mid in seen:
                    continue
                seen.add(mid)
                picks.append(r)
                reasons.append("random_baseline")

    if not picks:
        return df.iloc[0:0].copy(), reasons
    out = pd.DataFrame(picks).reset_index(drop=True)
    return out, reasons


def prepare_qe_validation(
    stage2_csv: Path,
    master_csv: Path,
    out_dir: Path,
    n_disagreement: int,
    n_consensus: int,
    protocol: QEProtocol,
    standardize_primitive: bool = True,
    n_random: int = 0,
    random_pool_csv: Optional[Path] = None,
    random_seed: int = 42,
    explicit_ids: Optional[List[str]] = None,
) -> int:
    df = pd.read_csv(stage2_csv)
    if df.empty:
        raise ValueError(f"Stage-2 CSV is empty: {stage2_csv}")

    if explicit_ids:
        subset, reasons = _select_rows_by_id(df, explicit_ids)
        if subset.empty:
            raise ValueError(
                f"None of the requested --ids were found in {stage2_csv}: {explicit_ids}"
            )
        LOG.warning("Explicit-ID mode: selected %d / %d requested material_ids",
                    len(subset), len(explicit_ids))
    else:
        random_pool_df: Optional[pd.DataFrame] = None
        if n_random > 0 and random_pool_csv is not None:
            random_pool_df = pd.read_csv(random_pool_csv)
            LOG.info("Random sampling pool: %s (%d rows)", random_pool_csv, len(random_pool_df))

        subset, reasons = _select_validation_rows(
            df,
            n_disagreement=n_disagreement,
            n_consensus=n_consensus,
            n_random=n_random,
            random_pool_df=random_pool_df,
            random_seed=random_seed,
        )
        if subset.empty:
            raise ValueError("No rows selected for QE validation; check stage-2 disagreement / consensus columns.")
        LOG.warning("Selected %d candidates for QE validation", len(subset))

    ids: Set[str] = set(subset["material_id"].astype(str))
    structs = load_structures_for_ids(master_csv, ids)
    if not structs:
        raise ValueError(f"Could not pull any structures from master corpus {master_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    protocol_payload = {"protocol": asdict(protocol), "stage2_csv": str(stage2_csv), "master_csv": str(master_csv)}
    (out_dir / "protocol.json").write_text(json.dumps(protocol_payload, indent=2), encoding="utf-8")

    written = 0
    index_rows: List[dict] = []
    for (_, row), reason in zip(subset.iterrows(), reasons):
        mid = str(row["material_id"])
        if mid not in structs:
            LOG.warning("Skipping %s — structure missing in master corpus", mid)
            continue
        s = structs[mid]
        if standardize_primitive:
            try:
                s = SpacegroupAnalyzer(s, symprec=1e-3).get_primitive_standard_structure()
            except Exception as exc:
                LOG.warning("SpacegroupAnalyzer failed for %s (%s); using raw structure", mid, exc)

        cand_dir = out_dir / mid
        cand_dir.mkdir(parents=True, exist_ok=True)

        scf_input, nscf_input = _build_pwinputs(s, protocol)
        scf_path = cand_dir / "scf.in"
        nscf_path = cand_dir / "nscf.in"
        scf_input.write_file(str(scf_path))
        nscf_input.write_file(str(nscf_path))

        f_syms = _f_block_symbols_in(s)
        plus_u_applied = bool(protocol.enable_plus_u_for_f and f_syms)
        if plus_u_applied:
            _inject_plus_u_and_spin(scf_path, f_syms,
                                    protocol.hubbard_u_eV,
                                    protocol.starting_magnetization_f)
            _inject_plus_u_and_spin(nscf_path, f_syms,
                                    protocol.hubbard_u_eV,
                                    protocol.starting_magnetization_f)

        s.to(filename=str(cand_dir / "POSCAR"), fmt="poscar")

        manifest = {
            "material_id": mid,
            "formula": str(row.get("formula", "")),
            "selection_reason": reason,
            "stage1_xgb_eV": float(row.get("stage1_xgb_eV", float("nan"))) if "stage1_xgb_eV" in row else None,
            "stage2_xgb_eV": float(row.get("stage2_xgb_eV", float("nan"))) if "stage2_xgb_eV" in row else None,
            "stage2_gnn_eV": float(row.get("stage2_gnn_eV", float("nan"))) if "stage2_gnn_eV" in row else None,
            "disagreement_eV": float(row.get("disagreement_eV", float("nan"))) if "disagreement_eV" in row else None,
            "agreement_class": str(row.get("agreement_class", "")),
            "kpoints_scf": list(_kpoint_grid_for_lattice(s, protocol.kpoint_density_per_inv_ang)),
            "kpoints_nscf": list(_kpoint_grid_for_lattice(s, protocol.nscf_kpoint_density_per_inv_ang)),
            "nbnd": _expected_nbnd(s, protocol.nbnd_padding),
            "n_atoms": int(len(s)),
            "lattice_abc": list(s.lattice.abc),
            "elements": sorted({site.specie.symbol for site in s}),
            "pseudo_map": _pseudo_map_for_structure(s),
            "f_block_symbols": f_syms,
            "plus_u_applied": plus_u_applied,
            "hubbard_u_eV": float(protocol.hubbard_u_eV) if plus_u_applied else None,
            "starting_magnetization_f": float(protocol.starting_magnetization_f) if plus_u_applied else None,
            "nspin": 2 if plus_u_applied else 1,
        }
        (cand_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        index_rows.append(manifest)
        written += 1

    (out_dir / "index.json").write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
    LOG.warning("Wrote %d QE candidate directories under %s", written, out_dir.as_posix())
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validation bridge: prepare rule-fixed Quantum ESPRESSO inputs for Stage-2 leads."
    )
    parser.add_argument(
        "--stage2",
        type=str,
        default=str(Path("results") / "deepesense_stage2_refined.csv"),
        help="Stage-2 refined CSV from compare_gnn_xgboost.py.",
    )
    parser.add_argument(
        "--master",
        type=str,
        default=str(Path("data") / "raw" / "deepesense_candidates.csv"),
        help="Master corpus CSV with serialized structures.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("dft_validation_qe")),
        help="Output directory for QE inputs.",
    )
    parser.add_argument("--n-disagreement", type=int, default=6, help="Number of strong-disagreement candidates.")
    parser.add_argument("--n-consensus", type=int, default=4, help="Number of finite-gap-consensus candidates.")
    parser.add_argument(
        "--n-random",
        type=int,
        default=0,
        help="Number of UNBIASED random samples (critical for paper-defensible precision @ K).",
    )
    parser.add_argument(
        "--random-pool",
        type=str,
        default=None,
        help="CSV to sample random rows from (defaults to --stage2). Pass --stage1 CSV here for an unbiased dark-matter-pool sample.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--no-primitive",
        action="store_true",
        help="Skip SpacegroupAnalyzer primitive-standard reduction (default: on).",
    )
    parser.add_argument("--ecutwfc", type=float, default=60.0, help="Plane-wave cutoff (Ry).")
    parser.add_argument("--ecutrho", type=float, default=480.0, help="Charge-density cutoff (Ry).")
    parser.add_argument("--kdens-scf", type=float, default=0.20, help="SCF k-point density (Å^-1).")
    parser.add_argument("--kdens-nscf", type=float, default=0.12, help="NSCF k-point density (Å^-1).")
    parser.add_argument(
        "--pseudo-family",
        type=str,
        default="SSSP_efficiency_v1.3_PBE",
        help="Tag stored in the protocol manifest; pseudo names are taken from the SSSP_PBE_PSEUDOS map.",
    )
    parser.add_argument(
        "--enable-plus-u",
        action="store_true",
        help="For structures containing a lanthanide/actinide, generate spin-polarised "
             "PBE+U inputs (nspin=2, starting_magnetization, Hubbard_U on the f species).",
    )
    parser.add_argument(
        "--hubbard-u",
        type=float,
        default=6.0,
        help="Hubbard U (eV) applied to the f-channel when --enable-plus-u is set. "
             "Default 6.0 eV is the canonical empirical value for Ln³⁺ ionic hosts.",
    )
    parser.add_argument(
        "--starting-mag-f",
        type=float,
        default=0.5,
        help="starting_magnetization for the f-block species (fraction in [-1, +1]). "
             "Only used when --enable-plus-u is set.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of material_id values to prepare (bypasses disagreement / "
             "consensus / random selection). Order is preserved.",
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        default=None,
        help="Optional newline-separated file of material_ids; merged with --ids.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()
    _configure_logging(args.verbose)

    explicit_ids: List[str] = []
    if args.ids:
        explicit_ids.extend(str(x) for x in args.ids)
    if args.ids_file:
        for line in Path(args.ids_file).read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                explicit_ids.append(s)

    protocol = QEProtocol(
        pseudo_family=args.pseudo_family,
        ecutwfc_Ry=float(args.ecutwfc),
        ecutrho_Ry=float(args.ecutrho),
        kpoint_density_per_inv_ang=float(args.kdens_scf),
        nscf_kpoint_density_per_inv_ang=float(args.kdens_nscf),
        enable_plus_u_for_f=bool(args.enable_plus_u),
        hubbard_u_eV=float(args.hubbard_u),
        starting_magnetization_f=float(args.starting_mag_f),
    )
    prepare_qe_validation(
        stage2_csv=Path(args.stage2),
        master_csv=Path(args.master),
        out_dir=Path(args.out),
        n_disagreement=int(args.n_disagreement),
        n_consensus=int(args.n_consensus),
        protocol=protocol,
        standardize_primitive=not args.no_primitive,
        n_random=int(args.n_random),
        random_pool_csv=Path(args.random_pool) if args.random_pool else None,
        random_seed=int(args.random_seed),
        explicit_ids=explicit_ids or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
