import os
from pathlib import Path
import pandas as pd

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter





def _read_xes(p: Path) -> pd.DataFrame:
    log = xes_importer.apply(str(p))
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    return df

# The data folder is in the project directory by default but feel free to change when you call load_file function
def _default_data_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "data").resolve()

def load_file(dataset: str, data_root: Path | None = None) -> pd.DataFrame:
    
    if data_root is None:
        data_root = _default_data_root()
    path = data_root / dataset
    if not path.exists():
        raise FileNotFoundError(f"no dataset found, check the path: {path}")

    # XES first
    xess  = sorted(path.glob("*.xes"))
    if xess:
        raw = _read_xes(xess[0])
        return raw

    csvs  = sorted(path.glob("*.csv"))
    xlsxs = sorted([*path.glob("*.xlsx"), *path.glob("*.xls")])

    if csvs:
        raw = pd.read_csv(csvs[0])
    elif xlsxs:
        raw = pd.read_excel(xlsxs[0])
    else:
        raise FileNotFoundError(f"No .xes/.csv/.xlsx found in: {path}")
    
    
    return raw
