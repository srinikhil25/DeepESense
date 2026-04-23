"""
Parse Quantum ESPRESSO ``pw.x`` outputs from the validation bridge.

Expected layout (created by ``qe_validation_prepare.py`` plus a real cluster run)::

    dft_validation_qe/
      protocol.json
      <material_id>/
        scf.in
        scf.out          <- written by `pw.x -in scf.in > scf.out`
        nscf.in
        nscf.out         <- written by `pw.x -in nscf.in > nscf.out`
        manifest.json

For each candidate the parser tries the NSCF output first (denser k-grid
makes the gap more reliable) and falls back to SCF if NSCF is missing or
unparseable. It tracks SCF convergence, total energy, magnetization, walltime,
and a structured ``gap_status`` for paper-grade diagnostics:

- ``ok``                 — finite gap extracted cleanly
- ``metallic``           — bands cross / overlap → recomputed gap is zero
- ``odd_electron_count`` — odd nelec without nspin=2; gap not well defined
- ``no_eigenvalues``     — neither the explicit gap line nor per-k blocks found
- ``unparseable``        — file present but parser could not read it
- ``missing_output``     — no ``*.out`` file at the expected path

Outputs:

- ``results/qe_validation_results.csv``
- ``results/qe_validation_results.csv.meta.json`` (status breakdown,
  parser version, gap tolerance used)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

LOG = logging.getLogger("deepesense.qe_validation_parse")

GAP_TOL_eV = 1e-3  # treat |LUMO - HOMO| < this as zero gap (matches Stage-1 convention)
PARSER_VERSION = "qe_validation_parse/1.0"


@dataclass
class PWResult:
    material_id: str
    source_file: str
    calculation: str  # 'scf' | 'nscf' | 'unknown'
    converged: Optional[bool]
    n_scf_iterations: Optional[int]
    total_energy_Ry: Optional[float]
    fermi_eV: Optional[float]
    homo_eV: Optional[float]
    lumo_eV: Optional[float]
    recomputed_pbe_gap_eV: Optional[float]
    gap_status: str
    gap_source: str  # 'explicit_line' | 'eigenvalues' | 'none'
    nelec: Optional[float]
    n_kpoints: Optional[int]
    n_bands: Optional[int]
    total_magnetization: Optional[float]
    wall_seconds: Optional[float]


# --- regexes ----------------------------------------------------------------

_RE_NELEC = re.compile(r"number of electrons\s*=\s*([\d.]+)")
_RE_NBND = re.compile(r"number of Kohn-Sham states\s*=\s*(\d+)")
_RE_FERMI = re.compile(r"the Fermi energy is\s+([-\d.]+)\s*ev")
_RE_HOMO_LUMO = re.compile(
    r"highest occupied,\s*lowest unoccupied level\s*\(ev\):\s+([-\d.]+)\s+([-\d.]+)"
)
_RE_HOMO_ONLY = re.compile(r"highest occupied level\s*\(ev\):\s+([-\d.]+)")
_RE_CONV = re.compile(r"convergence has been achieved in\s+(\d+) iterations")
_RE_NOT_CONV = re.compile(r"convergence NOT achieved", re.IGNORECASE)
_RE_TOTAL_E = re.compile(r"!\s*total energy\s*=\s*([-\d.]+)\s*Ry")
_RE_MAG = re.compile(r"total magnetization\s*=\s*([-\d.]+)")
_RE_CALC_NSCF = re.compile(r"calculation\s*=\s*['\"]nscf['\"]", re.IGNORECASE)
_RE_CALC_SCF = re.compile(r"calculation\s*=\s*['\"]scf['\"]", re.IGNORECASE)
_RE_WALL = re.compile(
    r"PWSCF\s*:.*?(?:(\d+)h\s*)?(?:(\d+)m\s*)?([\d.]+)s\s*WALL",
    re.DOTALL,
)
# k-point eigenvalue blocks. Captures everything between `bands (ev):` and
# the next k-point or terminator.
_RE_KBLOCK = re.compile(
    r"k\s*=\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\(\s*\d+\s*PWs\)\s*bands\s*\(ev\):\s*\n(.*?)(?=\n\s*k\s*=|\n\s*the Fermi|\n\s*highest occupied|\n\s*Writing|\n\s*occupation numbers|\Z)",
    re.DOTALL,
)
_RE_FLOAT = re.compile(r"-?\d+\.\d+")


def _parse_wall_seconds(text: str) -> Optional[float]:
    m = _RE_WALL.search(text)
    if not m:
        return None
    h = int(m.group(1)) if m.group(1) else 0
    mn = int(m.group(2)) if m.group(2) else 0
    s = float(m.group(3))
    return h * 3600 + mn * 60 + s


def _parse_eigenvalue_blocks(text: str) -> List[List[float]]:
    """Return a list of per-k eigenvalue lists from a pw.x output."""
    blocks: List[List[float]] = []
    for m in _RE_KBLOCK.finditer(text):
        chunk = m.group(1)
        floats = [float(x) for x in _RE_FLOAT.findall(chunk)]
        if floats:
            blocks.append(floats)
    return blocks


def _gap_from_blocks(
    eigs_per_k: List[List[float]], nelec: float
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """Return (homo, lumo, gap, status) using the standard insulator HOMO/LUMO rule."""
    if not eigs_per_k:
        return None, None, None, "no_eigenvalues"

    n_occ = nelec / 2.0
    if abs(n_occ - round(n_occ)) > 1e-6:
        # Non-spin-polarized calculation with odd electron count is unphysical
        # for the gap question; flag and skip.
        return None, None, None, "odd_electron_count"
    n_occ_int = int(round(n_occ))
    if n_occ_int < 1:
        return None, None, None, "no_eigenvalues"

    homos: List[float] = []
    lumos: List[float] = []
    for eigs in eigs_per_k:
        if len(eigs) < n_occ_int:
            continue
        homos.append(eigs[n_occ_int - 1])
        if len(eigs) >= n_occ_int + 1:
            lumos.append(eigs[n_occ_int])

    if not homos or not lumos:
        return None, None, None, "no_eigenvalues"

    homo = max(homos)
    lumo = min(lumos)
    gap = lumo - homo
    if gap < GAP_TOL_eV:
        return homo, lumo, max(gap, 0.0), "metallic"
    return homo, lumo, gap, "ok"


def parse_pwout(path: Path, material_id: str) -> PWResult:
    """Parse a single pw.x stdout file. Robust to missing / partial files."""
    if not path.exists():
        return PWResult(
            material_id=material_id,
            source_file=str(path),
            calculation="unknown",
            converged=None,
            n_scf_iterations=None,
            total_energy_Ry=None,
            fermi_eV=None,
            homo_eV=None,
            lumo_eV=None,
            recomputed_pbe_gap_eV=None,
            gap_status="missing_output",
            gap_source="none",
            nelec=None,
            n_kpoints=None,
            n_bands=None,
            total_magnetization=None,
            wall_seconds=None,
        )

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        LOG.warning("Could not read %s: %s", path, exc)
        return PWResult(
            material_id=material_id,
            source_file=str(path),
            calculation="unknown",
            converged=None,
            n_scf_iterations=None,
            total_energy_Ry=None,
            fermi_eV=None,
            homo_eV=None,
            lumo_eV=None,
            recomputed_pbe_gap_eV=None,
            gap_status="unparseable",
            gap_source="none",
            nelec=None,
            n_kpoints=None,
            n_bands=None,
            total_magnetization=None,
            wall_seconds=None,
        )

    if _RE_CALC_NSCF.search(text):
        calc = "nscf"
    elif _RE_CALC_SCF.search(text):
        calc = "scf"
    else:
        calc = "unknown"

    converged: Optional[bool] = None
    if _RE_CONV.search(text):
        converged = True
    elif _RE_NOT_CONV.search(text):
        converged = False

    n_iter = None
    m = _RE_CONV.search(text)
    if m:
        n_iter = int(m.group(1))

    total_e = None
    m = _RE_TOTAL_E.search(text)
    if m:
        total_e = float(m.group(1))

    fermi = None
    m = _RE_FERMI.search(text)
    if m:
        fermi = float(m.group(1))

    nelec = None
    m = _RE_NELEC.search(text)
    if m:
        nelec = float(m.group(1))

    nbnd = None
    m = _RE_NBND.search(text)
    if m:
        nbnd = int(m.group(1))

    mag = None
    m = _RE_MAG.search(text)
    if m:
        mag = float(m.group(1))

    wall = _parse_wall_seconds(text)

    homo: Optional[float] = None
    lumo: Optional[float] = None
    gap: Optional[float] = None
    status = "unparseable"
    gap_source = "none"

    # Priority 1: explicit "highest occupied, lowest unoccupied" line.
    m = _RE_HOMO_LUMO.search(text)
    if m:
        homo = float(m.group(1))
        lumo = float(m.group(2))
        gap = lumo - homo
        status = "ok" if gap >= GAP_TOL_eV else "metallic"
        gap_source = "explicit_line"

    # Priority 2: per-k eigenvalue parsing.
    blocks = _parse_eigenvalue_blocks(text)
    n_k = len(blocks) if blocks else None
    if status == "unparseable":
        if nelec is not None and blocks:
            homo, lumo, gap, status = _gap_from_blocks(blocks, nelec)
            if status in ("ok", "metallic"):
                gap_source = "eigenvalues"
        elif not blocks:
            status = "no_eigenvalues"

    return PWResult(
        material_id=material_id,
        source_file=str(path),
        calculation=calc,
        converged=converged,
        n_scf_iterations=n_iter,
        total_energy_Ry=total_e,
        fermi_eV=fermi,
        homo_eV=homo,
        lumo_eV=lumo,
        recomputed_pbe_gap_eV=gap,
        gap_status=status,
        gap_source=gap_source,
        nelec=nelec,
        n_kpoints=n_k,
        n_bands=nbnd,
        total_magnetization=mag,
        wall_seconds=wall,
    )


def _merge_scf_into(nscf: PWResult, scf: PWResult) -> PWResult:
    """When NSCF is missing convergence/energy/magnetization but SCF has them, lift them in."""
    if nscf.converged is None:
        nscf.converged = scf.converged
    if nscf.n_scf_iterations is None:
        nscf.n_scf_iterations = scf.n_scf_iterations
    if nscf.total_energy_Ry is None:
        nscf.total_energy_Ry = scf.total_energy_Ry
    if nscf.total_magnetization is None:
        nscf.total_magnetization = scf.total_magnetization
    if nscf.wall_seconds is None:
        nscf.wall_seconds = scf.wall_seconds
    return nscf


def parse_validation_dir(
    qe_dir: Path,
    out_csv: Path,
    scf_name: str = "scf.out",
    nscf_name: str = "nscf.out",
) -> pd.DataFrame:
    if not qe_dir.exists():
        raise FileNotFoundError(f"QE validation dir does not exist: {qe_dir}")

    candidates = sorted(p for p in qe_dir.iterdir() if p.is_dir())
    LOG.warning("Parsing %d candidate directories under %s", len(candidates), qe_dir)

    rows: List[dict] = []
    status_counts: Dict[str, int] = {}
    for cand in candidates:
        mid = cand.name
        nscf_path = cand / nscf_name
        scf_path = cand / scf_name

        nscf_result = parse_pwout(nscf_path, mid)
        scf_result = parse_pwout(scf_path, mid)

        # Combine: prefer NSCF for the gap, but lift convergence/energy/etc
        # from SCF when NSCF didn't capture them (NSCF runs don't iterate).
        if nscf_result.gap_status in ("ok", "metallic"):
            result = _merge_scf_into(nscf_result, scf_result)
        elif scf_result.gap_status in ("ok", "metallic"):
            result = scf_result
            # If we had a partial NSCF, surface its kpoint count
            if nscf_result.n_kpoints:
                result.n_kpoints = nscf_result.n_kpoints
        elif nscf_result.gap_status != "missing_output":
            # NSCF present but no gap; keep its diagnostics
            result = _merge_scf_into(nscf_result, scf_result)
        else:
            # Both missing or only SCF present
            result = scf_result

        # Read manifest for ML predictions cross-reference
        manifest_path = cand / "manifest.json"
        manifest: Dict = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                LOG.warning("Could not read manifest %s: %s", manifest_path, exc)

        row = asdict(result)
        for k in (
            "formula",
            "selection_reason",
            "stage1_xgb_eV",
            "stage2_xgb_eV",
            "stage2_gnn_eV",
            "disagreement_eV",
            "agreement_class",
        ):
            row[k] = manifest.get(k)
        rows.append(row)
        status_counts[result.gap_status] = status_counts.get(result.gap_status, 0) + 1

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    n_ok = int((df["gap_status"] == "ok").sum())
    n_metal = int((df["gap_status"] == "metallic").sum())
    summary = {
        "parser_version": PARSER_VERSION,
        "qe_dir": str(qe_dir),
        "n_candidate_dirs": len(candidates),
        "n_with_finite_gap": n_ok,
        "n_metallic": n_metal,
        "n_converged": int((df["converged"] == True).sum()),  # noqa: E712
        "status_breakdown": status_counts,
        "gap_tol_eV": GAP_TOL_eV,
        "scf_name": scf_name,
        "nscf_name": nscf_name,
    }
    out_csv.with_suffix(out_csv.suffix + ".meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOG.warning(
        "Parsed %d candidates: %d finite gap, %d metallic → %s",
        len(candidates),
        n_ok,
        n_metal,
        out_csv.as_posix(),
    )
    return df


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def main() -> int:
    p = argparse.ArgumentParser(description="Parse Quantum ESPRESSO pw.x outputs from the validation bridge.")
    p.add_argument(
        "--qe-dir",
        type=str,
        default=str(Path("dft_validation_qe")),
        help="Validation directory created by qe_validation_prepare.py.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(Path("results") / "qe_validation_results.csv"),
        help="Output results CSV.",
    )
    p.add_argument("--scf-name", type=str, default="scf.out")
    p.add_argument("--nscf-name", type=str, default="nscf.out")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    _configure_logging(args.verbose)
    parse_validation_dir(Path(args.qe_dir), Path(args.out), args.scf_name, args.nscf_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
