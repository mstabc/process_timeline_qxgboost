conda env create -f environment.yml
conda activate eventlogrepair



# Preprocess only (XES-first; reads DATA_DIR/<dataset>/*.xes)
python run.py --dataset <dataset_name> --preprocess
python run.py --dataset 2020_Permit_log --preprocess

python run.py --dataset 2012_A --preprocess --simulate-missing
python run.py --dataset 2017_O --preprocess --simulate-missing
python run.py --dataset 2020_Domestic_declarations --preprocess --simulate-missing
python run.py --dataset 2020_International_declarations --preprocess --simulate-missing
python run.py --dataset 2020_Prepaid_travel_cost --preprocess --simulate-missing
python run.py --dataset 2020_Request_for_payment --preprocess --simulate-missing
python run.py --dataset 2020_Request_for_payment --preprocess --simulate-missing
python run.py --dataset roadtraffic --preprocess --simulate-missing




# Train sequence models (start/complete quantiles)
python run.py --dataset <dataset_name> --sequence
python run.py --dataset 2012_A --sequence
python run.py --dataset 2017_O --sequence
python run.py --dataset 2020_Domestic_declarations --sequence
python run.py --dataset 2020_International_declarations --sequence
python run.py --dataset 2020_Prepaid_travel_cost --sequence
python run.py --dataset 2020_Request_for_payment --sequence
python run.py --dataset roadtraffic --sequence
python run.py --dataset sepsis --sequence


# Train duration models (start/complete quantiles)
python run.py --dataset <dataset_name> --duration
python run.py --dataset 2017_O --duration
python run.py --dataset 2020_Domestic_declarations --duration
python run.py --dataset 2020_International_declarations --duration
python run.py --dataset 2020_Prepaid_travel_cost --duration
python run.py --dataset 2020_Request_for_payment --duration
python run.py --dataset roadtraffic --duration
python run.py --dataset sepsis --duration


# Data Imputation