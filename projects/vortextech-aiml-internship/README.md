# VortexTech AI/ML Internship

This repository contains two educational Titanic notebooks:

- `week1.ipynb` performs descriptive data inspection and plots.
- `week2.ipynb` trains a deterministic decision-tree baseline with leakage-safe preprocessing.

## Data

The Titanic dataset is **not committed** because it may be subject to size or licensing constraints. A local `train.csv` is required, with at least `Survived`, `Age`, `Embarked`, and `Sex` columns. The notebooks fail early with a clear message when the file or required schema is missing; no data or metrics are fabricated.

Set `TITANIC_TRAIN_CSV` to an absolute path (recommended):

```bash
export TITANIC_TRAIN_CSV=/absolute/path/to/train.csv
```

If the variable is unset, execution from the repository root uses `data/titanic/train.csv`.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/validate_static.py
jupyter notebook
```

Run notebook cells from top to bottom. `week2.ipynb` splits before fitting imputers and one-hot encoding, uses a stratified split, and fixes `random_state=42`. It excludes identifiers/free text from this simple baseline and reports accuracy and F1 only after the held-out evaluation.

## Validation and limitations

- `python scripts/validate_static.py` checks Python syntax, notebook JSON/code syntax, execution counts, and committed outputs without requiring the dataset or ML dependencies.
- The repository currently contains no dataset, so a full notebook execution and numerical metrics cannot be verified here.
- Metrics depend on the input dataset, split, random seed, and library versions; they are not a production benchmark.
- These notebooks are educational artifacts, not medical or other professional decision systems.
