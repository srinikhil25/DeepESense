# DeepESense — QE Validation Evaluation

> **Claim under evaluation.** A non-trivial fraction of MP entries labelled `band_gap = 0.0 eV` are actually finite-gap when re-computed with a rule-fixed `pw.x` protocol; the two-stage ML pipeline concentrates them in the top-K above cheap baselines.

## 1. Validation set composition

- Total candidates parsed: **11**
- Status `ok` (clean finite gap): 2
- Status `metallic` (zero gap): 2
- Status `failed` (unparseable / missing / odd-electron): 7
- **Finite gap above threshold (0.05 eV): 1**

### Selection reason breakdown

| Reason | N | Finite-gap rate |
|---|---:|---:|
| finite_gap_consensus | 4 | 0.250 |
| strong_disagreement | 6 | 0.000 |

## 2. Base rates

- **overall_validated_set**: 0.091
- **finite_gap_consensus**: 0.250
- **strong_disagreement**: 0.000

> The `random_baseline` row above is the only **unbiased** estimate of the underlying base rate; the disagreement / consensus rows are conditional on the selection rule.

## 3. Regression metrics on finite-gap subset

Computed only on rows where `gap_status == 'ok'`.

| Method | N | MAE (eV) | RMSE (eV) | Spearman ρ |
|---|---:|---:|---:|---:|
| Stage 1 (XGBoost) | 2 | 2.922 | 3.888 | — |
| Stage 2 (XGBoost re-eval) | 2 | 2.922 | 3.888 | — |
| Stage 2 (GNN) | 2 | 1.851 | 2.073 | — |

## 4. Precision @ K (truth = recomputed PBE gap > 0.05 eV)

| Method | K=1 | K=3 | K=5 | K=10 | K=15 | K=20 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 1 (XGBoost) | 1.000 | 0.333 | 0.200 | 0.100 | 0.100 | 0.100 |
| Stage 2 (XGBoost re-eval) | 1.000 | 0.333 | 0.200 | 0.100 | 0.100 | 0.100 |
| Stage 2 (GNN) | 0.000 | 0.000 | 0.200 | 0.100 | 0.100 | 0.100 |
| Baseline: random | 0.000 | 0.000 | 0.000 | 0.100 | 0.091 | 0.091 |
| Baseline: max ΔEN | 0.000 | 0.000 | 0.000 | 0.100 | 0.091 | 0.091 |
| Baseline: n_elements | 0.000 | 0.000 | 0.000 | 0.100 | 0.091 | 0.091 |

![Precision @ K](reports/figures/qe_validation/precision_at_k.png)

## 5. Disagreement-set sanity

Of strong-disagreement rows where both Stage-2 models produced a prediction and the recomputed gap is known:

- **GNN closer to truth**: 2
- **XGB closer to truth**: 0
- Ties: 0
- Total in this slice: 2

![Disagreement winner](reports/figures/qe_validation/disagreement_winner.png)

## 6. Calibration

![Calibration](reports/figures/qe_validation/calibration.png)

## 7. Scatter plots (predicted vs recomputed)

### Stage 1 (XGBoost)

![Stage 1 (XGBoost)](reports/figures/qe_validation/scatter_stage1_xgb_eV.png)

### Stage 2 (XGBoost re-eval)

![Stage 2 (XGBoost re-eval)](reports/figures/qe_validation/scatter_stage2_xgb_eV.png)

### Stage 2 (GNN)

![Stage 2 (GNN)](reports/figures/qe_validation/scatter_stage2_gnn_eV.png)

## 8. Failure inspection — top mispredictions

| material_id | formula | selection_reason | recomputed_pbe_gap_eV | stage2_gnn_eV | stage2_xgb_eV | stage1_xgb_eV | abs_err_gnn |
|---|---|---|---|---|---|---|---|
| mp-1182442 | BO2F3 | strong_disagreement | 0.001 | 2.787 | 5.488 | 5.488 | 2.785 |
| mp-1185301 | LiF | finite_gap_consensus | 7.907 | 6.990 | 7.549 | 7.549 | 0.917 |

## 9. Caveats

- **Conditional precision.** Disagreement-driven and consensus-driven
  precisions are conditional on the selection rule. Only rows with
  `selection_reason = random_baseline` give an unbiased estimate of the
  underlying base rate. Quote conditional and unbiased numbers separately
  in the paper.
- **PBE vs PBE oracle.** The validation oracle is the same theory level as
  the training labels. This validates the *database-hygiene* claim ('MP
  zero-gap labels are unreliable') but not 'PBE was wrong about the
  electronic structure'. HSE / GW would be needed for the latter.
- **Magnetism.** The rule-fixed protocol does not enable spin polarization.
  Materials with magnetic ground states may be flagged metallic spuriously
  and will appear in the failure inspection table above.
- **Sample size.** With small N, precision @ K curves have high variance;
  bootstrap or pre-register the K values for the paper headline numbers.
