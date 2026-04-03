# DeepESense

DeepESense is an AI-driven pipeline for **unsupervised discovery of novel electronic materials**, starting from *candidate structures* that are theoretically stable but electronically under-characterized (“electronically dark”).

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root (do not commit it). You can copy from `.env.example`:

```bash
copy .env.example .env
```

Then edit `.env` and set `MP_API_KEY`.

## Full data acquisition (master dataset)

This fetches all Materials Project entries with \(E_{hull} < 0.05\) eV/atom in API batches and writes:

- `data/raw/deepesense_master_v1.csv`
- `logs/data_summary.txt`

Run full acquisition:

```bash
python -m src.discovery_engine -v
```

Use custom batching/checkpoint intervals:

```bash
python -m src.discovery_engine --batch-size 1000 --save-every 1000 -v
```

Test with a cap (optional):

```bash
python -m src.discovery_engine --max-records 5000 -v
```

## Feature engineering (matminer)

Generate compositional (Magpie) and structural (`DensityFeatures`) descriptors for the full master dataset:

```bash
python -m src.featurizer --in data/raw/deepesense_master_v1.csv --out data/processed/deepesense_features_v1.csv --n-jobs -1 -v
```

## Validation / QA (pictorial proof)

Validate `data/raw/mp_candidates_v1.csv`, remove invalid rows, generate plots and a Markdown report:

```bash
python -m src.verify_data -v
```

Outputs:

- `data/processed/mp_candidates_v1_clean.csv`
- `reports/deepesense_validation_report_v1.md`
- `reports/figures/validation_v1/*.png`

## Crystal GNN bandgap (PyTorch Geometric)

Install PyTorch and PyG first (see [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)), for example:

```bash
pip install torch torch-geometric
```

Train a CGCNN-style graph model on labeled materials (`bandgap > 0`) from `deepesense_master_v1.csv` (uses structure JSON + **5.0 Å** radius edges, **Schneider-style** loss that penalizes underestimating high bandgaps):

```bash
python -m src.train_gnn --master data/raw/deepesense_master_v1.csv --out models/gnn_bandgap_v1.pt -v
```

### Hardware tuning (e.g. RTX 4060 8GB VRAM + 8GB system RAM)

Defaults are already conservative (**`--batch-size 8`**, **`--epochs 30`**, **`--max-neighbors 32`**). If you hit **CUDA out-of-memory**, lower VRAM use in this order:

1. **`--batch-size 4`** or **`2`** (smallest change).
2. **`--accum-steps 4`** to keep an effective larger batch without more VRAM.
3. **`--max-neighbors 24`** or **`16`** (fewer edges per node).
4. **`--hidden 96`** or **`64`** (smaller model).
5. **`--empty-gpu-cache`** after each epoch to reduce fragmentation.

Example safe recipe for 8GB VRAM:

```bash
python -m src.train_gnn --batch-size 4 --accum-steps 4 --max-neighbors 24 --hidden 96 --epochs 30 --empty-gpu-cache -v
```

**System RAM:** All ~44k graphs are held in memory during training; 8GB RAM can be tight. Use **`--max-samples 10000`** for a shorter run, or train on a machine with more RAM, or we can add on-disk graph caching later.

Compare **GNN vs XGBoost** on the top dark-matter discoveries:

```bash
python -m src.compare_gnn_xgboost --discoveries results/deepesense_discoveries_v1.csv --gnn-ckpt models/gnn_bandgap_v1.pt --xgb-model models/baseline_xgboost_v1.json --features data/processed/deepesense_features_v1.csv -v
```

Writes `results/compare_gnn_xgboost_top10_v1.csv`.
