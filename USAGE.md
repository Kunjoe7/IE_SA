# ICDM-SA Package Usage Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/Kingmaoqin/ICDM-SA.git
cd ICDM-SA

# Install the package in development mode
pip install -e .
```

## Basic Usage

### 1. Using Built-in Datasets

```python
from icdm_sa import FLCHAINModel

# Create model instance
model = FLCHAINModel()

# Prepare data (automatic preprocessing)
train_loader, test_loader = model.prepare_data()

# Train the model
trainer = model.train()

# Evaluate
results = model.evaluate()
print(f"C-index: {results['c_index_test']:.4f}")
```

### 2. Using Your Own Data

```python
import pandas as pd
import torch
from icdm_sa import MultiTaskModel, EGTrainer, MultiTaskDataset

# Load your survival data
df = pd.read_csv('your_data.csv')
# Required columns: features, survival_time, event_indicator

# Preprocess data (see examples/example_custom_data.py)
# ... preprocessing steps ...

# Create model
model = MultiTaskModel(n_features, n_intervals)

# Train
trainer = EGTrainer(model, train_loader, test_loader, train_dataset, args)
trainer.train()

# Get predictions
trainer.load_best_checkpoint()
predictions, Y_hat, Y_true, events = trainer.predict(test_loader)
```

### 3. Configuring Training Parameters

```python
import easydict

args = easydict.EasyDict({
    "batch_size": 64,
    "cuda": True,  # Use GPU
    "lr": 0.01,
    "epochs": 200,
    "clip": 5.0,
    "lambda_reg": 0.01,
    "save_path": "results",
    "eg_k": 1,
    "early_stop_patience": 11,
})

model = FLCHAINModel(args=args)
```

## Advanced Features

### Feature Attribution

The model includes expected gradient attribution for interpretability:

```python
# After training
attribution_matrix = trainer.attribution_matrix
# This contains feature importance scores for each sample
```

### Model Persistence

```python
# Save model
torch.save(trainer.model.state_dict(), 'model.pth')

# Load model
model = MultiTaskModel(n_features, n_intervals)
model.load_state_dict(torch.load('model.pth'))
```

## Available Datasets

- **FLCHAIN**: Free Light Chain study (n=7,874)
- **GABS**: German AML Study 
- **METABRIC**: Breast cancer molecular data
- **NWTCO**: National Wilms Tumor study
- **TCGA_Task3**: The Cancer Genome Atlas

## Examples

See the `examples/` directory for complete working examples:
- `example_flchain.py`: Complete FLCHAIN analysis
- `example_custom_data.py`: Using your own survival data

## Citation

If you use this package in your research, please cite:

```
@article{icdmsa2024,
  title={Interactive and Explainable Survival Analysis},
  author={ICDM-SA Team},
  journal={ICDM},
  year={2024}
}
```