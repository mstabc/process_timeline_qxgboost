# EventLogRepair Conda Environment Setup

This guide explains how to create and use the **EventLogRepair** environment for the PyPOTS-based process mining Event log Repair.

---

## ⚙️ 1. Prerequisites

- **Anaconda or Miniconda** installed  
  → Download: [https://conda.io/miniconda.html](https://conda.io/miniconda.html)

- **NVIDIA GPU drivers** installed with CUDA 12.6 support  
  → Confirm by running:
  ```bash
  nvidia-smi


After saving both files:
```bash
conda env create -f environment.yml
conda activate eventlogrepair



# Preprocess only (XES-first; reads DATA_DIR/<dataset>/*.xes)
python run.py --dataset <dataset_name> --preprocess

# Train sequence models (start/complete quantiles)
python run.py --dataset <dataset_name> --train

# Do both
python run.py --dataset <dataset_name> --preprocess --train

# Optional: more logs
python run.py --dataset <dataset_name> --preprocess --train --verbose


