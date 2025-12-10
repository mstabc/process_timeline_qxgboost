conda env create -f environment.yml
conda activate eventlogrepair



# Preprocess only (XES-first; reads DATA_DIR/<dataset>/*.xes)
python run.py --dataset <dataset_name> --preprocess
python run.py --dataset 2020_Permit_log --preprocess


# Train sequence models (start/complete quantiles)
python run.py --dataset <dataset_name> --sequence
python run.py --dataset 2020_Permit_log --sequence


# Train duration models (start/complete quantiles)
python run.py --dataset <dataset_name> --duration


# Data Imputation