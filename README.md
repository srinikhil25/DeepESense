# DeepESense

A two-stage ML + DFT validation pipeline for re-ranking Materials Project
entries labelled `band_gap = 0.0 eV` as plausible finite-gap semiconductors.

> **Scope.** This is a portfolio / engineering piece that demonstrates a full
> materials-informatics workflow end-to-end: database pull → featurisation →
> tabular baseline → structure-aware GNN → disagreement-driven selection →
> first-principles validation with Quantum ESPRESSO → evaluation against
> cheap baselines. It is **not** a finished research paper; see
> [Limitations](#limitations) for honest scope.

---

## The question

PBE DFT is a known under-binder of band gaps. In Materials Project, a large
number of inorganic compounds ship with `band_gap = 0.0 eV`. Some of those
are genuine metals; others are narrow/mid-gap semiconductors that PBE
collapsed — *electronic dark matter*.

Can a cheap ML pipeline surface the dark-matter candidates in the top-K of a
ranking, more reliably than featureless baselines, and does independent DFT
confirm the hits?

## Pipeline

```
┌───────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ Stage 1 — scale   │   │ Stage 2 — fidelity   │   │ Validation bridge    │
│ XGBoost on Magpie │ → │ CGCNN-style GNN      │ → │ Rule-fixed QE SCF +  │
│ + DensityFeatures │   │ re-predicts gap from │   │ NSCF on disagreement │
│ over the full     │   │ actual crystal       │   │ / consensus picks    │
│ PBE=0 pool        │   │ structure            │   │                      │
└───────────────────┘   └──────────────────────┘   └──────────────────────┘
```

| Stage | Module | Output |
|---|---|---|
| Acquisition | `src/discovery_engine.py` | `data/raw/deepesense_candidates.csv` |
| Featurisation | `src/featurizer.py` | `data/processed/deepesense_features.csv` |
| Tabular trainer | `src/train_baseline.py` | `models/baseline_xgboost.json` |
| GNN trainer | `src/train_gnn.py` | `models/cgcnn.pt` |
| Stage 1 unfold | `src/discovery_inference.py` | `results/deepesense_dark_matter_unfolded.csv` |
| Stage 2 refine | `src/compare_gnn_xgboost.py` | `results/deepesense_stage2_refined.csv` |
| QE prepare | `src/qe_validation_prepare.py` | `dft_validation_qe/<mid>/scf.in`, `nscf.in` |
| QE parse | `src/qe_validation_parse.py` | `results/qe_validation_results.csv` |
| QE evaluate | `src/qe_validation_evaluate.py` | `results/qe_validation_report.md` + figures |

Every stage writes a `.meta.json` sidecar (model paths, thresholds,
imputer medians, random seeds, row counts) so every downstream number
traces back to a specific model + config.

---

## Results (N = 10 QE runs)

Tiny validation set — treat everything below as a demonstration of the
workflow, not a statistical claim.

### What QE confirmed

| Material | MP label | QE (PBE, this repo) | XGB pred | GNN pred | Verdict |
|---|---:|---:|---:|---:|---|
| **LiF** (mp-1185301) | 0.0 eV | **7.91 eV** | 7.55 | 6.99 | ✅ ML found a real mislabelled gap |
| **BO₂F₃** (mp-1182442) | 0.0 eV | 0.001 eV | 5.49 | 2.79 | ✅ True negative; high XGB–GNN disagreement was diagnostic |

### Selection-rule winners

| Metric | Value |
|---|---|
| XGBoost Precision@K=1 | **1.000** (top pick = LiF, truly gapped) |
| Disagreement-slice head-to-head | **GNN 2–0 over XGB** |
| Cheap baselines (random, max ΔEN, n_elements) @ K≤5 | 0.000 |

### What the protocol couldn't score (6/10)

Three f-electron materials (RbHoBeF6, KYb₃F₁₀, NiF₂) came back numerically
metallic because the current protocol is **non-spin-polarised** — a known
limitation for 4f open shells and antiferromagnets. Three more (NaPrF4,
BaTbF6, Li₂TbF₆, Li₄TbF₈) either failed to converge (f-electron charge
sloshing at β = 0.40) or were queued and not completed before the run
window ended.

See `results/qe_validation_report.md` for the full table, scatter plots,
precision@K curves, and calibration figure.

---

## Limitations

**Honest framing — what this demonstrates vs. what it doesn't claim.**

1. **Sample size.** N=10 attempted QE runs, 2 cleanly evaluable. Any
   regression metric (MAE, RMSE, ρ) on this is noise. Precision@K=1 is a
   single data point.
2. **Non-spin-polarised PBE.** The QE protocol assumes closed-shell,
   non-magnetic ground states. For f-electron lanthanides and transition
   metal fluorides this is wrong; those materials need `nspin=2` (and ideally
   +U or hybrid / meta-GGA) to be scored fairly. Running them as-is
   produced spurious metals.
3. **PBE vs. truth.** Even converged QE gaps are still PBE — they systematically
   underestimate real gaps. A proper validation set would use TBmBJ or
   hybrid functionals, or cross-check against JARVIS-DFT.
4. **Selection bias.** 4 of the 10 candidates were drawn from a
   *strong-disagreement* slice, which is exactly where ML should be most
   wrong. The base-rate number (0.091) is from a random slice of 11 and is
   too small to trust.
5. **No experimental anchor.** Every claim is "DFT says" vs. "DFT says."
   No measured gaps in the validation set.

**What I'd change for a real paper** (discussed but not done here):

- Pull JARVIS-DFT (~80k materials) and use TBmBJ gaps as a labelled
  validation set → scales N from 10 to a few hundred for free.
- Re-run the f-electron subset with `nspin=2` enabled in the protocol.
- Build a calibration plot for `|ΔXGB − ΔGNN|` as a label-noise predictor
  (ROC/AUC), which is the only part of this that might be novel.

---

## Reproduction

### Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate
pip install -r requirements.txt
# PyTorch + PyG: follow https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
echo "MP_API_KEY=<your-mp-api-key>" > .env
```

### Run order

```bash
# 1. Acquisition — full MP pull, e_hull < 0.05 eV/atom
python -m src.discovery_engine -v

# 2. Featurisation — Magpie + DensityFeatures
python -m src.featurizer \
    --in data/raw/deepesense_candidates.csv \
    --out data/processed/deepesense_features.csv -v

# 3. Train models
python -m src.train_baseline -v
python -m src.train_gnn -v

# 4. Stage 1 + Stage 2 ranking
python -m src.discovery_inference -v
python -m src.compare_gnn_xgboost --top-k 50 -v

# 5. Prepare QE inputs for disagreement + consensus + random picks
python -m src.qe_validation_prepare \
    --stage2 results/deepesense_stage2_refined.csv \
    --out dft_validation_qe \
    --n-disagreement 6 --n-consensus 4 --n-random 10 -v

# --- manual: run pw.x scf.in then nscf.in inside each dft_validation_qe/<mid>/ ---
# See dft_validation_qe/run_sequential.sh for the actual driver used here.

# 6. Parse + evaluate
python -m src.qe_validation_parse --qe-dir dft_validation_qe \
    --out results/qe_validation_results.csv -v
python -m src.qe_validation_evaluate \
    --in results/qe_validation_results.csv -v
```

### DFT protocol (rule-fixed)

| Setting | Value |
|---|---|
| Functional | PBE |
| Pseudopotentials | SSSP Efficiency v1.3 (PBE) |
| `ecutwfc` / `ecutrho` | 60 / 480 Ry |
| Smearing | Marzari–Vanderbilt, 0.005 Ry |
| `conv_thr` | 1e-8 |
| `electron_maxstep` | 200 |
| Mixing `beta` | 0.40 |
| k-points | Γ-centred, density 0.2 Å⁻¹ |
| Spin | Non-polarised (limitation — see above) |

Full protocol: `dft_validation_qe/protocol.json`.

### Hardware notes (GNN training)

8 GB VRAM is tight. If OOM, tune in order:
`--batch-size 4` → `--accum-steps 4` → `--max-neighbors 24` →
`--hidden 96` → `--empty-gpu-cache`.

Safe recipe:
```bash
python -m src.train_gnn --batch-size 4 --accum-steps 4 \
    --max-neighbors 24 --hidden 96 --epochs 30 --empty-gpu-cache -v
```

---

## Repo layout

```
src/                      pipeline modules (each runnable as python -m src.<mod>)
  discovery_engine.py       Materials Project pull
  featurizer.py             Magpie + DensityFeatures
  train_baseline.py         XGBoost
  train_gnn.py              CGCNN-style GNN
  holdout_split.py          reproducible train/val/test split
  discovery_inference.py    Stage 1 ranking
  compare_gnn_xgboost.py    Stage 2 refinement + disagreement tagging
  qe_validation_prepare.py  writes rule-fixed QE inputs
  qe_validation_parse.py    walks QE outputs → structured CSV
  qe_validation_evaluate.py metrics, figures, markdown report
scripts/smoke_qe_validation.py  synthetic end-to-end smoke test
dft_validation_qe/        actual QE run dir (N=10)
  protocol.json, run_sequential.sh, sequential_run.log
  mp-*/                     per-material scf.in/out, nscf.in/out, manifest.json
results/                  CSVs + evaluation markdown report
reports/figures/          baseline parity plot + QE evaluation figures
models/                   trained XGBoost + CGCNN checkpoints
```

Large regenerable artefacts (`data/raw/`, `data/processed/`, personal
`logs/`) are gitignored. Run `src/discovery_engine.py` + `src/featurizer.py`
to rebuild them.
