import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import easydict
from torch.utils.data import DataLoader
import os
from pathlib import Path

from ...algorithm.model_imp import MultiTaskModel
from ...algorithm.multi_task_dataset import MultiTaskDataset
from ...algorithm.expected_gradient_trainer import EGTrainer
from ...algorithm import unique_value_counts, Cindex, brier_score

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FLCHAINModel:
    def __init__(self, data_path=None, args=None):
        """
        Initialize FLCHAIN survival analysis model
        
        Args:
            data_path: Path to flchain.csv file. If None, uses packaged data
            args: EasyDict with model parameters. If None, uses defaults
        """
        set_seed(1)
        
        # Default arguments
        self.args = args or easydict.EasyDict({
            "batch_size": 64,
            "cuda": torch.cuda.is_available(),
            "lr": 0.01,
            "epochs": 200,
            "clip": 5.0,
            "lambda_reg": 0.01,
            "save_path": "outputfiles",
            "eg_k": 1,
            "early_stop_patience": 11,
        })
        
        # Load data
        if data_path is None:
            # Use packaged data
            current_dir = Path(__file__).parent
            data_path = current_dir / "flchain.csv"
        
        self.df = pd.read_csv(data_path)
        self.preprocess_data()
        
    def preprocess_data(self):
        """Preprocess the FLCHAIN dataset"""
        data = self.df
        
        # Remove instances with survival time <= 0
        data = data.drop(data[data['futime'] <= 0].index)
        data.dropna(inplace=True)
        
        self.data_original = data.copy(deep=True)
        print(f"Data shape after preprocessing: {data.shape}")
        print(f"Survival time min: {data.futime.min()}, max: {data.futime.max()}")
        
        # Create intervals
        data_sorted = data.sort_values(by='futime').reset_index(drop=True)
        bin_size = 150
        data_sorted['interval_number'] = data_sorted.index // bin_size + 1
        
        self.num_intervals = len(data_sorted['interval_number'].unique())
        print(f"Number of intervals: {self.num_intervals}")
        
        # Map intervals back to original data
        data['interval_number'] = data['futime'].apply(
            lambda st: data_sorted[data_sorted['futime'] == st]['interval_number'].iloc[0]
        )
        
        # Create label and mask vectors
        data['label_vector'] = data['interval_number'].apply(self._label_vector)
        data['mask_vector'] = data.apply(
            lambda x: self._mask_vector(x.interval_number, x.death), axis=1
        )
        
        self.processed_data = data
        
    def _label_vector(self, interval_number):
        lv = np.zeros(self.num_intervals)
        lv[:interval_number] = 1
        return lv.tolist()
    
    def _mask_vector(self, interval_number, event):
        if event == 1:
            mv = np.ones(self.num_intervals)
            return mv.tolist()
        else:
            mv = np.zeros(self.num_intervals)
            mv[:interval_number] = 1
            return mv.tolist()
    
    def prepare_data(self, test_size=0.25, random_state=1):
        """Prepare data for training"""
        data = self.processed_data
        
        # Split indices
        train_indices, test_indices = train_test_split(
            range(len(data)), 
            stratify=data['death'], 
            random_state=random_state, 
            test_size=test_size
        )
        
        # Prepare features and labels
        X = data.drop(['futime', 'death', 'interval_number', 'label_vector', 'mask_vector'], axis=1)
        Y = np.array(data['label_vector'].values.tolist())
        W = np.array(data['mask_vector'].values.tolist())
        events = data['death'].values
        
        self.X_columns = X.columns
        
        # Scale features
        scaler = MinMaxScaler()
        X_train = X.values[train_indices]
        X_test = X.values[test_indices]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to tensors
        Y_train = torch.Tensor(Y[train_indices])
        Y_test = torch.Tensor(Y[test_indices])
        W_train = torch.Tensor(W[train_indices])
        W_test = torch.Tensor(W[test_indices])
        X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
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
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=True)
        
        self.train_dataset = train_dataset
        self.n_features = X_train.shape[1]
        
        return self.train_loader, self.test_loader
    
    def train(self):
        """Train the model"""
        # Create save directory
        Path(self.args.save_path).mkdir(exist_ok=True, parents=True)
        
        # Initialize model
        self.model = MultiTaskModel(self.n_features, self.num_intervals)
        
        # Initialize trainer
        self.trainer = EGTrainer(
            self.model, 
            self.train_loader, 
            self.test_loader, 
            self.train_dataset, 
            self.args
        )
        
        # Train
        self.trainer.train()
        
        return self.trainer
    
    def evaluate(self):
        """Evaluate the model and return C-index scores"""
        cindex_calculator = Cindex()
        
        # Load best checkpoint
        self.trainer.load_best_checkpoint()
        
        # Training set evaluation
        predictions, Y_hat, Y_true, events = self.trainer.predict(self.train_loader)
        c_train = cindex_calculator(Y_true, Y_hat, events)
        
        # Test set evaluation  
        predictions, Y_hat, Y_true, events = self.trainer.predict(self.test_loader)
        c_test = cindex_calculator(Y_true, Y_hat, events)
        
        return {
            'c_index_train': c_train,
            'c_index_test': c_test
        }