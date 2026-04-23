"""
Smoke test for src.qe_validation_parse + src.qe_validation_evaluate.

Constructs a synthetic dft_validation_qe directory with three candidates
covering the three interesting regimes:

  1. mp-100  — clean NSCF with finite gap (explicit HOMO/LUMO line)     → gap_status = ok
  2. mp-200  — metallic (HOMO/LUMO overlap) via per-k eigenvalue blocks → gap_status = metallic
  3. mp-300  — finite gap derived from per-k eigenvalue blocks
              (no explicit line; insulating)                             → gap_status = ok

Each candidate carries a manifest.json mimicking what qe_validation_prepare
writes so the evaluator has Stage-1 / Stage-2 ML predictions to cross-check.

Run:
    python scripts/smoke_qe_validation.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.qe_validation_parse import parse_validation_dir  # noqa: E402
from src.qe_validation_evaluate import evaluate  # noqa: E402


SCF_OUT_OK = """\
     Program PWSCF v.7.2 starts on ...

     calculation     = 'scf'

     number of electrons       =        16.00
     number of Kohn-Sham states=           24

     total cpu time spent up to now is      2.3 secs

     convergence has been achieved in  11 iterations

!    total energy              =    -250.12345678 Ry
     total magnetization       =     0.00 Bohr mag/cell

     the Fermi energy is     3.2100 ev

     PWSCF        :     3.12s CPU      3.45s WALL
"""

NSCF_OUT_OK = """\
     Program PWSCF v.7.2 starts on ...

     calculation     = 'nscf'

     number of electrons       =        16.00
     number of Kohn-Sham states=           24

          k = 0.0000 0.0000 0.0000 (  1234 PWs)   bands (ev):

  -10.12  -9.87  -8.55  -7.12  -5.01  -3.88  -2.11  -1.02
    3.45   4.12   5.03   6.11   7.22   8.14   9.05  10.11
   11.00  12.10  13.20  14.30  15.40  16.50  17.60  18.70

          k = 0.5000 0.0000 0.0000 (  1240 PWs)   bands (ev):

  -10.05  -9.80  -8.50  -7.05  -4.95  -3.80  -2.05  -0.95
    3.55   4.20   5.10   6.20   7.30   8.20   9.10  10.20
   11.10  12.20  13.30  14.40  15.50  16.60  17.70  18.80

     the Fermi energy is     1.2000 ev

     highest occupied, lowest unoccupied level (ev):    -0.9500    3.4500

     PWSCF        :     2.01s CPU      2.30s WALL
"""


SCF_OUT_METAL = """\
     Program PWSCF v.7.2 starts on ...

     calculation     = 'scf'

     number of electrons       =         8.00
     number of Kohn-Sham states=           16

     convergence has been achieved in   9 iterations

!    total energy              =    -120.98765432 Ry
     total magnetization       =     0.00 Bohr mag/cell

     the Fermi energy is     5.5000 ev

     PWSCF        :     1.45s CPU      1.62s WALL
"""

# Metallic: bands overlap — HOMO > LUMO conceptually, explicit line shows zero gap.
NSCF_OUT_METAL = """\
     Program PWSCF v.7.2 starts on ...

     calculation     = 'nscf'

     number of electrons       =         8.00
     number of Kohn-Sham states=           16

          k = 0.0000 0.0000 0.0000 (   900 PWs)   bands (ev):

   -8.10  -6.00  -4.20  -2.10   1.20   2.50   3.80   4.90
    5.60   6.70   7.80   8.90  10.00  11.00  12.00  13.00

          k = 0.5000 0.5000 0.0000 (   912 PWs)   bands (ev):

   -7.90  -5.80  -4.05   1.40   2.10   2.80   4.00   5.00
    5.70   6.80   7.90   9.00  10.10  11.10  12.10  13.10

     the Fermi energy is     2.3000 ev

     PWSCF        :     1.11s CPU      1.25s WALL
"""


SCF_OUT_INSULATOR = """\
     Program PWSCF v.7.2 starts on ...

     calculation     = 'scf'

     number of electrons       =        12.00
     number of Kohn-Sham states=           20

     convergence has been achieved in  15 iterations

!    total energy              =    -310.55555555 Ry
     total magnetization       =     0.00 Bohr mag/cell

     the Fermi energy is     1.8000 ev

     PWSCF        :     4.00s CPU      4.30s WALL
"""

# Insulator without explicit line; must derive gap from per-k eigenvalues.
# nelec=12 → 6 occupied bands per k. HOMO = max over k of band 6; LUMO = min over k of band 7.
NSCF_OUT_INSULATOR = """\
     Program PWSCF v.7.2 starts on ...

     calculation     = 'nscf'

     number of electrons       =        12.00
     number of Kohn-Sham states=           20

          k = 0.0000 0.0000 0.0000 (  1500 PWs)   bands (ev):

   -9.00  -7.50  -5.10  -3.20  -1.80  -0.50   1.20   2.80
    3.90   4.80   5.70   6.60   7.50   8.40   9.30  10.20
   11.10  12.00  12.90  13.80

          k = 0.5000 0.0000 0.0000 (  1512 PWs)   bands (ev):

   -8.90  -7.40  -5.00  -3.10  -1.70  -0.40   1.30   2.90
    4.00   4.90   5.80   6.70   7.60   8.50   9.40  10.30
   11.20  12.10  13.00  13.90

          k = 0.5000 0.5000 0.0000 (  1520 PWs)   bands (ev):

   -8.80  -7.30  -4.90  -3.00  -1.60  -0.30   1.40   3.00
    4.10   5.00   5.90   6.80   7.70   8.60   9.50  10.40
   11.30  12.20  13.10  14.00

     PWSCF        :     5.22s CPU      5.55s WALL
"""


CANDIDATES = [
    {
        "mid": "mp-100",
        "scf": SCF_OUT_OK,
        "nscf": NSCF_OUT_OK,
        "manifest": {
            "material_id": "mp-100",
            "formula": "SiO2",
            "selection_reason": "random_baseline",
            "stage1_xgb_eV": 2.9,
            "stage2_xgb_eV": 3.1,
            "stage2_gnn_eV": 4.8,
            "disagreement_eV": 1.7,
            "agreement_class": "strong",
        },
    },
    {
        "mid": "mp-200",
        "scf": SCF_OUT_METAL,
        "nscf": NSCF_OUT_METAL,
        "manifest": {
            "material_id": "mp-200",
            "formula": "Fe2O3",
            "selection_reason": "disagreement",
            "stage1_xgb_eV": 0.4,
            "stage2_xgb_eV": 0.2,
            "stage2_gnn_eV": 1.6,
            "disagreement_eV": 1.4,
            "agreement_class": "strong",
        },
    },
    {
        "mid": "mp-300",
        "scf": SCF_OUT_INSULATOR,
        "nscf": NSCF_OUT_INSULATOR,
        "manifest": {
            "material_id": "mp-300",
            "formula": "MgO",
            "selection_reason": "finite_gap_consensus",
            "stage1_xgb_eV": 1.5,
            "stage2_xgb_eV": 1.8,
            "stage2_gnn_eV": 1.9,
            "disagreement_eV": 0.1,
            "agreement_class": "agree",
        },
    },
]


def build_fixture(qe_dir: Path) -> None:
    if qe_dir.exists():
        shutil.rmtree(qe_dir)
    qe_dir.mkdir(parents=True)
    (qe_dir / "protocol.json").write_text(
        json.dumps({"smoke_test": True, "functional": "pbe"}, indent=2),
        encoding="utf-8",
    )
    for c in CANDIDATES:
        d = qe_dir / c["mid"]
        d.mkdir()
        (d / "scf.out").write_text(c["scf"], encoding="utf-8")
        (d / "nscf.out").write_text(c["nscf"], encoding="utf-8")
        (d / "manifest.json").write_text(json.dumps(c["manifest"], indent=2), encoding="utf-8")


def main() -> int:
    tmp_root = ROOT / "_smoke_qe"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir()

    qe_dir = tmp_root / "dft_validation_qe"
    build_fixture(qe_dir)

    out_csv = tmp_root / "results" / "qe_validation_results.csv"
    df = parse_validation_dir(qe_dir, out_csv)

    print("\n=== parsed rows ===")
    print(df[["material_id", "gap_status", "gap_source", "recomputed_pbe_gap_eV",
              "homo_eV", "lumo_eV", "converged", "n_scf_iterations"]].to_string(index=False))

    # Assertions on parser output
    row_100 = df[df["material_id"] == "mp-100"].iloc[0]
    row_200 = df[df["material_id"] == "mp-200"].iloc[0]
    row_300 = df[df["material_id"] == "mp-300"].iloc[0]

    assert row_100["gap_status"] == "ok", f"mp-100 expected ok, got {row_100['gap_status']}"
    assert row_100["gap_source"] == "explicit_line"
    assert abs(row_100["recomputed_pbe_gap_eV"] - 4.40) < 1e-3, row_100["recomputed_pbe_gap_eV"]
    assert row_100["converged"] is True or row_100["converged"] == True  # noqa: E712
    assert row_100["n_scf_iterations"] == 11

    assert row_200["gap_status"] == "metallic", f"mp-200 expected metallic, got {row_200['gap_status']}"
    assert row_200["recomputed_pbe_gap_eV"] is not None

    assert row_300["gap_status"] == "ok", f"mp-300 expected ok, got {row_300['gap_status']}"
    assert row_300["gap_source"] == "eigenvalues"
    # nelec=12 → 6 occupied; HOMO = max(-0.30, -0.40, -0.50) = -0.30; LUMO = min(1.20, 1.30, 1.40) = 1.20
    expected_gap = 1.20 - (-0.30)
    assert abs(row_300["recomputed_pbe_gap_eV"] - expected_gap) < 1e-3, row_300["recomputed_pbe_gap_eV"]

    print("\n[parse] all assertions passed.")

    # Now evaluate
    merged_csv = tmp_root / "results" / "qe_validation_merged.csv"
    figures_dir = tmp_root / "reports" / "figures" / "qe_validation_v1"
    report_md = tmp_root / "reports" / "qe_validation_evaluation_v1.md"

    merged = evaluate(
        qe_results_csv=out_csv,
        out_csv=merged_csv,
        figures_dir=figures_dir,
        report_path=report_md,
        truth_threshold_eV=0.05,
        ks=[1, 2, 3],
    )

    assert merged_csv.exists()
    assert report_md.exists()
    expected_figs = [
        "precision_at_k.png",
        "calibration.png",
        "disagreement_winner.png",
        "scatter_stage1_xgb_eV.png",
        "scatter_stage2_xgb_eV.png",
        "scatter_stage2_gnn_eV.png",
    ]
    for f in expected_figs:
        p = figures_dir / f
        assert p.exists() and p.stat().st_size > 0, f"figure missing or empty: {p}"

    print("[evaluate] merged CSV, report, and all figures exist.")
    print(f"[evaluate] report path: {report_md}")
    print(f"[evaluate] figures dir: {figures_dir}")

    # Sanity: truth_finite_gap should be True for mp-100 and mp-300, False for mp-200
    tf = merged.set_index("material_id")["truth_finite_gap"].to_dict()
    assert tf["mp-100"] is True or tf["mp-100"] == True  # noqa: E712
    assert tf["mp-300"] is True or tf["mp-300"] == True  # noqa: E712
    assert tf["mp-200"] is False or tf["mp-200"] == False  # noqa: E712

    # Clean up on success
    shutil.rmtree(tmp_root)
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
