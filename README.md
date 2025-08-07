# Interactive and Explainable Survival Analysis

This repository contains the ICDM-SA package for Interactive and Explainable Survival Analysis experiments.

## Features

- **Interactive survival analysis** with explainable AI techniques
- **Multiple pre-processed datasets**: FLCHAIN, GABS, METABRIC, NWTCO, TCGA_Task3
- **Multi-task learning** approach for survival prediction
- **Expected gradient attribution** for model interpretability
- **Easy-to-use API** for both pre-loaded and custom datasets

## Installation

### Install from source

```bash
git clone https://github.com/Kingmaoqin/ICDM-SA.git
cd IE_SA
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## Quick Start

### Using Pre-loaded Datasets

```python
from icdm_sa import FLCHAINModel

# Initialize and train model on FLCHAIN dataset
model = FLCHAINModel()
train_loader, test_loader = model.prepare_data()
trainer = model.train()

# Evaluate model
results = model.evaluate()
print(f"C-index (Test): {results['c_index_test']:.4f}")
```

### Using Custom Data

```python
import pandas as pd
from icdm_sa import MultiTaskModel, EGTrainer, MultiTaskDataset

# Load your data
df = pd.read_csv('your_survival_data.csv')

# Prepare data (see examples/example_custom_data.py for full example)
# ... data preprocessing ...

# Train model
model = MultiTaskModel(n_features, n_intervals)
trainer = EGTrainer(model, train_loader, test_loader, train_dataset, args)
trainer.train()
```

## Datasets

The package includes five pre-processed survival analysis datasets:

- **FLCHAIN**: Free Light Chain dataset
- **GABS**: German AML Study dataset  
- **METABRIC**: Molecular Taxonomy of Breast Cancer dataset
- **NWTCO**: National Wilms Tumor dataset
- **TCGA_Task3**: The Cancer Genome Atlas dataset

## Examples

See the `examples/` directory for detailed usage examples:
- `example_flchain.py`: Using the package with FLCHAIN dataset
- `example_custom_data.py`: Using the package with custom survival data

## Requirements

- **Python**: 3.8 or later (tested on 3.8–3.11)  
- **Key Packages**:  
  - `torch>=1.9.0`
  - `numpy>=1.19.0`  
  - `pandas>=1.3.0`  
  - `scikit-learn>=0.24.0`  
  - `matplotlib>=3.3.0`
  - `seaborn>=0.11.0`
  - `tqdm>=4.62.0`
  - `easydict>=1.9`
  - `lifelines>=0.27.0`  
  - `scipy>=1.7.0`  
