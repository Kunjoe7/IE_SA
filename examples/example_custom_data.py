"""
Example script showing how to use ICDM-SA with custom data
"""

import pandas as pd
import numpy as np
import torch
from icdm_sa import MultiTaskModel, EGTrainer, MultiTaskDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import easydict
from pathlib import Path

def prepare_custom_data(df, survival_col='time', event_col='event', bin_size=150):
    """
    Prepare custom survival data for ICDM-SA
    
    Args:
        df: DataFrame with survival data
        survival_col: Column name for survival time
        event_col: Column name for event indicator (0/1)
        bin_size: Size of time intervals
    """
    # Sort by survival time
    data_sorted = df.sort_values(by=survival_col).reset_index(drop=True)
    
    # Create intervals
    data_sorted['interval_number'] = data_sorted.index // bin_size + 1
    num_intervals = len(data_sorted['interval_number'].unique())
    
    # Map intervals back
    df['interval_number'] = df[survival_col].apply(
        lambda st: data_sorted[data_sorted[survival_col] == st]['interval_number'].iloc[0]
    )
    
    # Create label vectors
    def label_vector(interval_number):
        lv = np.zeros(num_intervals)
        lv[:interval_number] = 1
        return lv.tolist()
    
    # Create mask vectors
    def mask_vector(interval_number, event):
        if event == 1:
            mv = np.ones(num_intervals)
            return mv.tolist()
        else:
            mv = np.zeros(num_intervals)
            mv[:interval_number] = 1
            return mv.tolist()
    
    df['label_vector'] = df['interval_number'].apply(label_vector)
    df['mask_vector'] = df.apply(
        lambda x: mask_vector(x.interval_number, x[event_col]), axis=1
    )
    
    return df, num_intervals

def main():
    # Load your custom data
    # df = pd.read_csv('your_data.csv')
    
    # For demo, create synthetic survival data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate survival times and events
    baseline_hazard = 0.1
    linear_predictor = X[:, 0] * 0.5 + X[:, 1] * -0.3 + X[:, 2] * 0.2
    hazard = baseline_hazard * np.exp(linear_predictor)
    survival_times = np.random.exponential(1/hazard)
    censoring_times = np.random.exponential(10)
    
    observed_times = np.minimum(survival_times, censoring_times)
    events = (survival_times <= censoring_times).astype(int)
    
    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['time'] = observed_times
    df['event'] = events
    
    print(f"Generated {n_samples} samples with {events.sum()} events")
    
    # Prepare data for ICDM-SA
    df_processed, num_intervals = prepare_custom_data(df)
    print(f"Number of intervals: {num_intervals}")
    
    # Split data
    train_indices, test_indices = train_test_split(
        range(len(df_processed)), 
        stratify=df_processed['event'], 
        random_state=42, 
        test_size=0.25
    )
    
    # Prepare features and labels
    X = df_processed[feature_cols]
    Y = np.array(df_processed['label_vector'].values.tolist())
    W = np.array(df_processed['mask_vector'].values.tolist())
    events = df_processed['event'].values
    
    # Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X.values[train_indices])
    X_test = scaler.transform(X.values[test_indices])
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_train = torch.Tensor(Y[train_indices])
    Y_test = torch.Tensor(Y[test_indices])
    W_train = torch.Tensor(W[train_indices])
    W_test = torch.Tensor(W[test_indices])
    event_train = events[train_indices]
    event_test = events[test_indices]
    
    # Transform for multi-task learning
    Y_train_transform = [Y_train[:, i:i + 1] for i in range(Y_train.size(1))]
    Y_test_transform = [Y_test[:, i:i + 1] for i in range(Y_test.size(1))]
    W_train_transform = [W_train[:, i:i+1] for i in range(W_train.size(1))]
    W_test_transform = [W_test[:, i:i+1] for i in range(W_test.size(1))]
    
    # Create datasets and loaders
    train_dataset = MultiTaskDataset(X_train, Y_train_transform, W_train_transform, event_train)
    test_dataset = MultiTaskDataset(X_test, Y_test_transform, W_test_transform, event_test)
    
    # Configure training
    args = easydict.EasyDict({
        "batch_size": 32,
        "cuda": torch.cuda.is_available(),
        "lr": 0.01,
        "epochs": 50,
        "clip": 5.0,
        "lambda_reg": 0.01,
        "save_path": "custom_results",
        "eg_k": 1,
        "early_stop_patience": 11,
    })
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    # Create save directory
    Path(args.save_path).mkdir(exist_ok=True, parents=True)
    
    # Initialize and train model
    model = MultiTaskModel(n_features, num_intervals)
    trainer = EGTrainer(model, train_loader, test_loader, train_dataset, args)
    
    print("Training model...")
    trainer.train()
    
    # Evaluate
    from icdm_sa import Cindex
    cindex_calculator = Cindex()
    
    trainer.load_best_checkpoint()
    predictions, Y_hat, Y_true, events = trainer.predict(test_loader)
    c_index = cindex_calculator(Y_true, Y_hat, events)
    
    print(f"\nTest C-index: {c_index:.4f}")

if __name__ == "__main__":
    main()