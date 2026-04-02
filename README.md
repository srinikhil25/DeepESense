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

## Discovery: build a candidate-structure dataset (dry run default)

This fetches **stable-ish** ternary/quaternary materials (default \(E_{hull} \le 0.02\) eV/atom) that are **electronically under-characterized** (by default `has_props == False` in MP Summary). It then applies a **uniqueness filter** (rare element combinations + chemistry motifs) and writes:

- `data/raw/mp_candidates_v1.csv`

Run the default **dry run (10 results)**:

```bash
python -m src.discovery_engine -v
```

Run a full query (no limit):

```bash
python -m src.discovery_engine --no-limit -v
```

Change the stability threshold:

```bash
python -m src.discovery_engine --ehull-max 0.01 -v
```

Skip structure retrieval (composition-only workflow):

```bash
python -m src.discovery_engine --no-structures -v
```

Include materials that already have electronic properties (`has_props == True`):

```bash
python -m src.discovery_engine --allow-has-props -v
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
