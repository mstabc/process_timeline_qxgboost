# Process Timeline Pipeline

This repository contains an event-log pipeline for:
- preprocessing process-mining datasets,
- sequence prediction (`relative_start`, `relative_complete`),
- duration class prediction (`binary_duration_class`, `multi_duration_class`),
- duration quantile regression (task duration).

Imputation and missingness simulation components were removed.

## Environment

```bash
conda env create -f environment.yml
conda activate eventlogrepair
```

## Data Layout

Put each dataset under:

```text
data/<dataset_name>/
```

Supported formats inside each dataset folder:
- `.xes` (preferred),
- `.csv`,
- `.xlsx` / `.xls`.

## Run

Preprocess:

```bash
python run.py --dataset sepsis --preprocess
```

Train sequence models:

```bash
python run.py --dataset sepsis --sequence
```

Train duration classifiers + quantile regression:

```bash
python run.py --dataset sepsis --duration
```

Run full flow:

```bash
python run.py --dataset sepsis --preprocess --sequence --duration
```

## Outputs

Generated outputs are written under `outputs/` and ignored by Git.
