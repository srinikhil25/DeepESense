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

Generate compositional (Magpie) and structural descriptors for clustering/novelty detection:

```bash
python -m src.featurizer --in data/raw/mp_candidates_v1.csv --out data/processed/mp_candidates_features_v1.csv -v
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
