import importlib
import pandas as pd
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import easydict
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import trange
from torch.nn.utils import clip_grad_norm_
from algorithm.multi_task_dataset import MultiTaskDataset
from algorithm import (
                        # NoRegularizationTrainer,
                        #MultiTaskModel,
                        MultiTaskDataset,
                        mlt_train_test_split,
                        # true_values_from_data_loader,
                        unique_value_counts,
                        Cindex,
                        brier_score,
                        )
from algorithm.no_reg_trainer1 import NoRegularizationTrainer
from algorithm.expected_gradient_trainer import EGTrainer
import random
import os.path

import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def set_seed(seed=42):
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch (CPU)
    torch.manual_seed(seed)

    # Set the seed for PyTorch (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example usage
set_seed(1)

df = pd.read_csv('/project/lwang/xqin5/SAcompare/NWTCO/nwtco.csv')
data = df
# data = data[data["survival_time"] <= 4000]
# Removing instances with survival time <= 0
data = data.drop(data[data['edrel'] <= 0].index)
print((data.isna().sum()>0).any())
data.dropna(inplace=True)
# data = data.head(1000)
print("DATA SHAPE: ", data.shape)
data_original = data.copy(deep=True)

args = easydict.EasyDict({
    "batch_size": 64,
    "cuda": True, # should set it to be true when using gpu, otherwise data would be on two devices
    "lr": 0.01,
    "epochs": 200,
    "clip": 5.0,
    "lambda_reg": 0.01,
    "save_path": "outputfiles",
    "eg_k" : 1, 
    "early_stop_patience":11,
})

data = data_original.copy(deep=True)

print(f"Survival time min: {data.edrel.min()}, max: {data.edrel.max()}")




# Sort the DataFrame by the specified column
data_sorted = data.sort_values(by='edrel').reset_index(drop=True)

# Define the bin size
bin_size = 150 # 67 intervals
# bin_size = 200 # 50 intervals

# Assign bins to each instance
data_sorted['interval_number'] = data_sorted.index // bin_size + 1
# Verify the distribution
bins_distribution = data_sorted['interval_number'].value_counts().sort_index()
num_intervals = len(data_sorted['interval_number'].unique())
print("Num Intervals: ", num_intervals)
print(bins_distribution)
data_sorted.head(20)  # View the updated DataFrame

data['interval_number'] = data['edrel'].apply(
    lambda st: data_sorted[data_sorted['edrel'] == st]['interval_number'].iloc[0]
)

data_sorted = data.copy(deep=True)

def label_vector(interval_number):
    lv = np.zeros(num_intervals)
    lv[:interval_number] = 1
    return lv.tolist()

def mask_vector(interval_number, event):
    if event == 1:
        mv = np.ones(num_intervals)
        return  mv.tolist()
    else:
        mv = np.zeros(num_intervals)
        mv[:interval_number] = 1
        return  mv.tolist()

data_sorted["label_vector"] = data_sorted['interval_number'].apply(label_vector)
data_sorted["mask_vector"] = data_sorted.apply(lambda x: mask_vector(x.interval_number, x.rel), axis=1)




train_indices, test_indices = train_test_split(
        range(len(data_sorted)), stratify=data_sorted['rel'], random_state=1, test_size=0.25
    )

X = data_sorted.drop(['edrel', 'rel', 'interval_number', 'label_vector', 'mask_vector'], axis=1)
Y = np.array(data_sorted['label_vector'].values.tolist())
W = np.array(data_sorted['mask_vector'].values.tolist())
events = data_sorted['rel'].values

X_columns = X.columns

scaler = MinMaxScaler()
X_train = X.values[train_indices]
X_test = X.values[test_indices]
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)



Y_train = torch.Tensor(Y[train_indices])
Y_test = torch.Tensor(Y[test_indices])
W_train = torch.Tensor(W[train_indices])
W_test = torch.Tensor(W[test_indices])
# X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
# X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
event_train = events[train_indices]
event_test = events[test_indices]

Y_train_transform = [Y_train[:, i:i + 1] for i in range(Y_train.size(1))]
Y_test_transform = [Y_test[:, i:i + 1] for i in range(Y_test.size(1))]
W_train_transform = [W_train[:, i:i+1] for i in range(W_train.size(1))]
W_test_transform = [W_test[:, i:i+1] for i in range(W_test.size(1))]

print((W_test_transform[-1] == 0).any())

train_dataset = MultiTaskDataset(X_train, Y_train_transform, W_train_transform, event_train)
test_dataset = MultiTaskDataset(X_test, Y_test_transform, W_test_transform, event_test)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)






static_task_weights = []
# Compute static task weights based on the total number of samples per task
for i in range(len(Y_test_transform)):
    total_masked_samples = Y_test_transform[i].sum() + 1e-6  # Avoid division by zero
    static_task_weights.append(1.0 / total_masked_samples)
static_task_weights = torch.tensor(static_task_weights).cuda()
static_task_weights = static_task_weights / static_task_weights.sum()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path
Path(args.save_path).mkdir(exist_ok=True, parents=True)


class EditedCindexOptimized(torch.nn.Module):

    def __init__(self):

        super(EditedCindexOptimized, self).__init__()

    def forward(self, y, y_hat, status):
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32)
        if not torch.is_tensor(y_hat):
            y_hat = torch.tensor(y_hat, dtype=torch.float32)
        if not torch.is_tensor(status):
            status = torch.tensor(status, dtype=torch.float32)

        # replacing loop acceleration with matrix calculation

        y_diff = y.unsqueeze(1) - y.unsqueeze(0)
        y_hat_diff = y_hat.unsqueeze(1) - y_hat.unsqueeze(0)
        # status[i] and status[j] mark whether to censored data
        status_i = status.unsqueeze(1)
        status_j = status.unsqueeze(0)
        valid_pairs = torch.logical_or((y_diff < 0) & (status_i == 1), (y_diff > 0) & (status_j == 1)).float()
        torch.diagonal(valid_pairs).fill_(0)  #Diagonal set to 0 to eliminate interference
        concordant_pairs = torch.logical_or((y_diff < 0) & (y_hat_diff < 0) & (status_i == 1),
                                            (y_diff > 0) & (y_hat_diff > 0) & (status_j == 1)).float()
        torch.diagonal(concordant_pairs).fill_(0)  #Diagonal set to 0 to eliminate interference
        concordant_pairs = concordant_pairs.float()
        c_index = concordant_pairs.sum() / valid_pairs.sum()
        return c_index.item()

# Simplified Shared Layer for Tabular Data
import torch.nn as nn
import torch.nn.functional as F
class SimplifiedSharedLayer(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(SimplifiedSharedLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.layers(x)

# Main MultiTask Model
class MultiTaskModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=128):
        super(MultiTaskModel, self).__init__()
        self.num_tasks = out_features

        # Shared Layer simplified for tabular data
        self.shared_layer = SimplifiedSharedLayer(in_features, hidden_dim)
        
        # Task-specific layers
        self.task_layers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(out_features)])
        # self.task_layers = nn.ModuleList()
        # share_output_shape = hidden_dim
        # for _ in range(out_features):
        #     task_layer = nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.LeakyReLU(),
        #         nn.BatchNorm1d(hidden_dim),
        #         nn.Linear(hidden_dim, 1),
        #     )
        #     self.task_layers.append(task_layer)

    def forward(self, x):
        # Shared Feature Extraction
        shared_output = self.shared_layer(x)  # Shape: (batch_size, hidden_dim)
        # shared_output = torch.cat([x, shared_output], dim=1)
        # Task Outputs
        task_outputs = []
        for i, task_layer in enumerate(self.task_layers):
            if i == 0:
                task_output = torch.sigmoid(task_layer(shared_output))
            else:
                task_output = torch.sigmoid(task_layer(shared_output)) * task_outputs[-1]
            task_outputs.append(task_output)
            # task_outputs.append(torch.sigmoid(task_layer(shared_output)))

        return task_outputs

    # def custom_loss(self, task_outputs, targets, masks):
    #     classification_loss = 0
    #     for i in range(len(task_outputs)):
    #         task_output = task_outputs[i]
    #         task_target = targets[i]
    #         task_mask = masks[i]
    #         # Binary cross-entropy loss for each task
    #         task_loss = F.binary_cross_entropy(task_output, task_target.float(), reduction='none')
    #         task_loss = task_loss * task_mask
    #         # Weight each task's loss by its dynamic task weight
    #         weighted_task_loss = (task_loss.sum() / task_mask.sum()) * static_task_weights[i]
    #         classification_loss += weighted_task_loss
    #     return classification_loss
    
    # def custom_loss(self, task_outputs, targets, masks):
    #     classification_loss = 0
    #     dynamic_task_weights = []
    # 
    #     # Compute dynamic task weights based on the total number of samples per task
    #     for i in range(len(targets)):
    #         total_masked_samples = targets[i].sum() + 1e-6  # Avoid division by zero
    #         dynamic_task_weights.append(1.0 / total_masked_samples)
    # 
    #     dynamic_task_weights = torch.tensor(dynamic_task_weights).cuda()
    #     dynamic_task_weights = dynamic_task_weights / dynamic_task_weights.sum()  # Normalize weights
    #     for i in range(len(task_outputs)):
    #         task_output = task_outputs[i]
    #         task_target = targets[i]
    #         task_mask = masks[i]
    #         # Binary cross-entropy loss for each task
    #         task_loss = F.binary_cross_entropy(task_output, task_target.float(), reduction='none')
    #         task_loss = task_loss * task_mask
    #         # Weight each task's loss by its dynamic task weight
    #         weighted_task_loss = (task_loss.sum() / task_mask.sum()) * dynamic_task_weights[i]
    #         classification_loss += weighted_task_loss
    #     return classification_loss
    
    def custom_loss(self, task_outputs, targets, masks):
        loss = 0
        for i, task_output in enumerate(task_outputs):
            task_target = targets[i]
            task_mask = masks[i]
            task_loss = F.binary_cross_entropy(task_output, task_target.float(), reduction='none')
            task_loss = task_loss * task_mask.float()
            loss += task_loss.sum() / task_mask.sum()
        return loss
    

model = MultiTaskModel(X_train.shape[1], Y_train.shape[1])
optimizer = optim.NAdam(model.parameters(), lr=args.lr, weight_decay=1e-2)
# trainer = EGTrainer(model,train_loader,test_loader, train_dataset, args)
trainer = NoRegularizationTrainer(model,train_loader,test_loader, optimizer, args)
trainer.train()
# (64x20096 and 19968x100)
# model = model.MultiTaskModel(633, 10)
# model.forward(torch.rand((64, 633)))
cindex_calculator_optimized = EditedCindexOptimized()

trainer.load_best_checkpoint()
predictions, Y_hat, Y_true, events = trainer.predict(train_loader)
# Y_hat = predictions[np.arange(predictions.shape[0]), (Y_hat-1)]
c11_train = cindex_calculator_optimized(Y_true, Y_hat, events)
print(f"C-index for Training Data: {c11_train:.4f}")


predictions, Y_hat, Y_true, events = trainer.predict(test_loader)
# Y_hat = predictions[np.arange(predictions.shape[0]), (Y_hat-1)]
c11_test = cindex_calculator_optimized(Y_true, Y_hat, events)
print(f"C-index for Test Data: {c11_test:.4f}")



# gradient_importance = trainer.gradient_importance.cpu().numpy()
# feature_columns = X_columns

# # 按绝对值排序并取前10个
# top_n = 10
# sorted_indices_global = np.argsort(-np.abs(gradient_importance))[:top_n]
# top_global_gradients = gradient_importance[sorted_indices_global]
# top_global_columns = [feature_columns[i] for i in sorted_indices_global]

# # 绘制水平条形图
# plt.figure(figsize=(10, 8))
# plt.barh(top_global_columns, top_global_gradients, color=['#FF0051' if g > 0 else '#0051FF' for g in top_global_gradients])


# # 局部特征重要性绘图
# sample_idx = 0
# X_sample = X_train[sample_idx:sample_idx + 1].to(trainer.device)
# X_sample.requires_grad = True

# # 获取单个样本的梯度
# task_outputs_sample = trainer.model(X_sample)
# trainer.model.zero_grad()
# sample_loss = sum([output.sum() for output in task_outputs_sample])
# sample_loss.backward()
# local_gradients = X_sample.grad.cpu().numpy().flatten()

# # 按绝对值排序并取前10个
# sorted_indices_local = np.argsort(-np.abs(local_gradients))[:top_n]
# top_local_gradients = local_gradients[sorted_indices_local]
# top_local_columns = [feature_columns[i] for i in sorted_indices_local]

# # 绘制水平条形图
# plt.figure(figsize=(10, 8))
# plt.barh(top_local_columns, top_local_gradients, color=['#FF0051' if g > 0 else '#0051FF' for g in top_local_gradients])

# # 添加标签
# for index, value in enumerate(top_local_gradients):
#     plt.text(value, index, f'{value:.3f}', va='center', fontsize=8, color='black')

# plt.xlabel("Global Feature Importance Value")
# plt.title(f"Top 10 Global Feature Importance", fontsize=14, fontweight='bold')
# plt.tight_layout()

# # 保存图片
# output_file_local = os.path.join(trainer.args.save_path, f"top_10_global_feature_importance.png")
# plt.savefig(output_file_local, bbox_inches='tight')
# plt.show()
# print(f"Global feature importance plot saved to {output_file_local}")

