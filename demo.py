#!/usr/bin/env python
"""
Quick demo of ICDM-SA package with synthetic data
"""

import numpy as np
import torch
from icdm_sa.algorithm.model_imp import MultiTaskModel
from icdm_sa.algorithm.multi_task_dataset import MultiTaskDataset
from icdm_sa.algorithm.expected_gradient_trainer import EGTrainer
from icdm_sa.algorithm.cindex import Cindex
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import easydict
from pathlib import Path
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("ICDM-SA Demo with Synthetic Survival Data")
print("="*60)

# 1. Generate synthetic survival data
print("\n1. Generating synthetic survival data...")
n_samples = 500
n_features = 8

# Generate features
X = np.random.randn(n_samples, n_features)
feature_names = [f'Feature_{i+1}' for i in range(n_features)]

# Generate survival times based on features
# True coefficients for simulation
true_coefs = np.array([0.5, -0.3, 0.2, 0, 0, 0.4, -0.2, 0.1])
linear_predictor = X @ true_coefs
baseline_hazard = 0.1
hazard = baseline_hazard * np.exp(linear_predictor)

# Generate survival and censoring times
survival_times = np.random.exponential(1/hazard)
censoring_times = np.random.exponential(5)  # Independent censoring
observed_times = np.minimum(survival_times, censoring_times)
events = (survival_times <= censoring_times).astype(int)

print(f"  - Generated {n_samples} samples with {n_features} features")
print(f"  - Event rate: {events.mean():.2%}")
print(f"  - Median survival time: {np.median(observed_times):.2f}")

# 2. Prepare data for ICDM-SA
print("\n2. Preparing data for multi-task survival analysis...")

# Create time intervals
bin_size = 20
n_intervals = 10

# Create interval assignments
time_bins = np.linspace(0, observed_times.max(), n_intervals + 1)
interval_numbers = np.digitize(observed_times, time_bins)

# Create label and mask vectors
def create_label_vector(interval_num):
    labels = np.zeros(n_intervals)
    labels[:min(interval_num, n_intervals)] = 1
    return labels

def create_mask_vector(interval_num, event):
    if event == 1:
        masks = np.ones(n_intervals)
    else:
        masks = np.zeros(n_intervals)
        masks[:min(interval_num, n_intervals)] = 1
    return masks

Y = np.array([create_label_vector(i) for i in interval_numbers])
W = np.array([create_mask_vector(i, e) for i, e in zip(interval_numbers, events)])

print(f"  - Created {n_intervals} time intervals")
print(f"  - Time bins: {time_bins}")

# 3. Split and scale data
print("\n3. Splitting data into train/test sets...")
train_idx, test_idx = train_test_split(
    range(n_samples), test_size=0.25, stratify=events, random_state=42
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X[train_idx])
X_test = scaler.transform(X[test_idx])

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y[train_idx], dtype=torch.float32)
Y_test = torch.tensor(Y[test_idx], dtype=torch.float32)
W_train = torch.tensor(W[train_idx], dtype=torch.float32)
W_test = torch.tensor(W[test_idx], dtype=torch.float32)
events_train = events[train_idx]
events_test = events[test_idx]

# Prepare for multi-task learning
Y_train_list = [Y_train[:, i:i+1] for i in range(n_intervals)]
Y_test_list = [Y_test[:, i:i+1] for i in range(n_intervals)]
W_train_list = [W_train[:, i:i+1] for i in range(n_intervals)]
W_test_list = [W_test[:, i:i+1] for i in range(n_intervals)]

print(f"  - Train set: {len(train_idx)} samples")
print(f"  - Test set: {len(test_idx)} samples")

# 4. Create datasets and dataloaders
print("\n4. Creating PyTorch datasets and dataloaders...")
train_dataset = MultiTaskDataset(X_train, Y_train_list, W_train_list, events_train)
test_dataset = MultiTaskDataset(X_test, Y_test_list, W_test_list, events_test)

# Training configuration
args = easydict.EasyDict({
    "batch_size": 32,
    "cuda": torch.cuda.is_available(),
    "lr": 0.01,
    "epochs": 20,  # Reduced for demo
    "clip": 5.0,
    "lambda_reg": 0.01,
    "save_path": "demo_results",
    "eg_k": 1,
    "early_stop_patience": 5,
})

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

# 5. Create and train model
print("\n5. Creating and training the model...")
print(f"  - Using device: {'cuda' if args.cuda else 'cpu'}")

Path(args.save_path).mkdir(exist_ok=True, parents=True)
model = MultiTaskModel(n_features, n_intervals)
trainer = EGTrainer(model, train_loader, test_loader, train_dataset, args)

print("\nTraining progress:")
trainer.train()

# 6. Evaluate model
print("\n6. Evaluating model performance...")
cindex = Cindex()
trainer.load_best_checkpoint()

# Training set performance
predictions_train, Y_hat_train, Y_true_train, events_train_pred = trainer.predict(train_loader)
c_index_train = cindex(Y_true_train, Y_hat_train, events_train_pred)

# Test set performance
predictions_test, Y_hat_test, Y_true_test, events_test_pred = trainer.predict(test_loader)
c_index_test = cindex(Y_true_test, Y_hat_test, events_test_pred)

print(f"\nModel Performance:")
print(f"  - Training C-index: {c_index_train:.4f}")
print(f"  - Test C-index: {c_index_test:.4f}")

# 7. Feature importance (if available)
print("\n7. Analyzing feature importance...")
if hasattr(trainer, 'attribution_matrix') and trainer.attribution_matrix is not None:
    # Get global feature importance
    global_importance = trainer.attribution_matrix.mean(axis=0).cpu().numpy()
    
    # Sort features by importance
    importance_idx = np.argsort(np.abs(global_importance))[::-1]
    
    # Plot top features
    plt.figure(figsize=(10, 6))
    top_k = min(8, n_features)
    top_features = importance_idx[:top_k]
    top_importance = global_importance[top_features]
    top_names = [feature_names[i] for i in top_features]
    
    colors = ['red' if x > 0 else 'blue' for x in top_importance]
    plt.bar(range(top_k), top_importance, color=colors)
    plt.xticks(range(top_k), top_names, rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Global Importance')
    plt.title('Feature Importance for Survival Prediction')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{args.save_path}/feature_importance.png')
    print(f"  - Feature importance plot saved to {args.save_path}/feature_importance.png")
    
    # Print feature importance
    print("\n  Top Features by Importance:")
    for i, (feat_idx, imp) in enumerate(zip(top_features, top_importance)):
        true_coef = true_coefs[feat_idx]
        print(f"    {i+1}. {feature_names[feat_idx]}: {imp:+.4f} (true coef: {true_coef:+.2f})")
else:
    print("  - Feature attribution not available")

# 8. Make predictions on new data
print("\n8. Making predictions on new samples...")
# Generate a few new samples
new_samples = torch.randn(3, n_features)
new_samples_scaled = torch.tensor(scaler.transform(new_samples.numpy()), dtype=torch.float32)

model.eval()
with torch.no_grad():
    survival_probs = model(new_samples_scaled.to(trainer.device))
    
print("  Survival probabilities for 3 new samples:")
for i in range(3):
    probs = [prob[i].item() for prob in survival_probs]
    print(f"    Sample {i+1}: {[f'{p:.3f}' for p in probs[:5]]}... (first 5 intervals)")

print("\n" + "="*60)
print("Demo completed successfully!")
print("="*60)
print("\nThe package is working correctly and can be used for:")
print("  - Multi-task survival analysis")
print("  - Feature importance analysis") 
print("  - Survival probability prediction")
print("\nFor more examples, see the examples/ directory.")