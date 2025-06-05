import importlib
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
import os
import seaborn as sns
import random
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
from algorithm.model_imp import MultiTaskModel
###############################################################################
# 1. Set random seed for reproducibility
###############################################################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)

###############################################################################
# 2. Load and preprocess data (NWTCO dataset)
###############################################################################
df = pd.read_csv('/project/lwang/xqin5/Armin_TCGI/cancer_data_633_gene.csv')
data = df
# data = df.head(1000)

data = data[data["survival_time"] <= 4000]
# Removing instances with survival time <= 0
data = data.drop(data[data['survival_time'] <= 0].index)
print((data.isna().sum()>0).any())
data.dropna(inplace=True)
# data = data.head(1000)
print("DATA SHAPE: ", data.shape)
data_original = data.copy(deep=True)


args = easydict.EasyDict({
    "batch_size": 64,
    "cuda": True,
    "lr": 0.001,        
    "epochs": 200,
    "clip": 5.0,
    "lambda_reg": 0.01,
    "save_path": "outputfiles",
    "eg_k": 1,
    "early_stop_patience": 11,
    "lambda_smooth": 0.01,   
    "steps_ig": 20,         
    "g_lr": 1e-4,           
    "d_lr": 1e-4, 
    "lambda_1": 0.01,
    "lambda_2": 0.01
})

print(f"Survival time min: {data.survival_time.min()}, max: {data.survival_time.max()}")

###############################################################################
# 2.1 Create intervals and label/mask vectors
###############################################################################

# Sort the DataFrame by the specified column
data_sorted = data.sort_values(by='survival_time').reset_index(drop=True)

# Define the bin size
bin_size = 50 # 67 intervals
# bin_size = 200 # 50 intervals

# Assign bins to each instance
data_sorted['interval_number'] = data_sorted.index // bin_size + 1
# Verify the distribution
bins_distribution = data_sorted['interval_number'].value_counts().sort_index()
num_intervals = len(data_sorted['interval_number'].unique())
print("Num Intervals: ", num_intervals)
print(bins_distribution)
data_sorted.head(20)  # View the updated DataFrame

data['interval_number'] = data['survival_time'].apply(
    lambda st: data_sorted[data_sorted['survival_time'] == st]['interval_number'].iloc[0]
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
data_sorted["mask_vector"] = data_sorted.apply(lambda x: mask_vector(x.interval_number, x.indicater), axis=1)

train_indices, test_indices = train_test_split(
        range(len(data_sorted)), stratify=data_sorted['indicater'], random_state=1, test_size=0.25
    )

X = data_sorted.drop(['survival_time', 'indicater', 'interval_number', 'label_vector', 'mask_vector'], axis=1)
Y = np.array(data_sorted['label_vector'].values.tolist())
W = np.array(data_sorted['mask_vector'].values.tolist())
events = data_sorted['indicater'].values

X_columns = X.columns
scaler = MinMaxScaler()
X_train = X.values[train_indices]
X_test = X.values[test_indices]
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Y_train = torch.Tensor(Y[train_indices])
Y_test = torch.Tensor(Y[test_indices])
W_train = torch.Tensor(W[train_indices])
W_test = torch.Tensor(W[test_indices])
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
event_train = events[train_indices]
event_test = events[test_indices]

Y_train_transform = [Y_train[:, i:i + 1] for i in range(Y_train.size(1))]
Y_test_transform = [Y_test[:, i:i + 1] for i in range(Y_test.size(1))]
W_train_transform = [W_train[:, i:i+1] for i in range(W_train.size(1))]
W_test_transform = [W_test[:, i:i+1] for i in range(W_test.size(1))]

print((W_test_transform[-1] == 0).any())

###############################################################################
# 2.2 MultiTaskDataset
###############################################################################
class MultiTaskDataset(Dataset):
    def __init__(self, X, Y_list, M_list, events):
        self.X = X
        self.Y_list = Y_list
        self.M_list = M_list
        self.events = events

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.X[idx],
            [y[idx] for y in self.Y_list],
            [m[idx] for m in self.M_list],
            self.events[idx],
        )

train_dataset = MultiTaskDataset(X_train, Y_train_transform, W_train_transform, event_train)
test_dataset = MultiTaskDataset(X_test, Y_test_transform, W_test_transform, event_test)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

###############################################################################
# 3. MultiTaskModel
###############################################################################
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

# class MultiTaskModel(nn.Module):
#     def __init__(self, in_features, out_features, hidden_dim=128):
#         super(MultiTaskModel, self).__init__()
#         self.num_tasks = out_features
#         self.shared_layer = SimplifiedSharedLayer(in_features, hidden_dim)
#         self.task_layers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(out_features)])

#     def forward(self, x):
#         shared_output = self.shared_layer(x)
#         task_outputs = []
#         for i, task_layer in enumerate(self.task_layers):
#             if i == 0:
#                 task_output = torch.sigmoid(task_layer(shared_output))
#             else:
#                 task_output = torch.sigmoid(task_layer(shared_output)) * task_outputs[-1]
#             task_outputs.append(task_output)
#         return task_outputs

#     def custom_loss(self, task_outputs, targets, masks):
#         loss = 0
#         for i, task_output in enumerate(task_outputs):
#             t_target = targets[i]
#             t_mask   = masks[i]
#             t_loss   = F.binary_cross_entropy(task_output, t_target.float(), reduction='none')
#             t_loss   = t_loss * t_mask.float()
#             loss    += t_loss.sum() / (t_mask.sum() + 1e-6)
#         return loss

###############################################################################
# 4. Simplified c-index
###############################################################################
class EditedCindexOptimized(nn.Module):
    def __init__(self):
        super(EditedCindexOptimized, self).__init__()
    def forward(self, y, y_hat, status):
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32)
        if not torch.is_tensor(y_hat):
            y_hat = torch.tensor(y_hat, dtype=torch.float32)
        if not torch.is_tensor(status):
            status = torch.tensor(status, dtype=torch.float32)
        y_diff = y.unsqueeze(1) - y.unsqueeze(0)
        y_hat_diff = y_hat.unsqueeze(1) - y_hat.unsqueeze(0)
        status_i = status.unsqueeze(1)
        status_j = status.unsqueeze(0)

        valid_pairs = torch.logical_or((y_diff < 0) & (status_i == 1),
                                       (y_diff > 0) & (status_j == 1)).float()
        torch.diagonal(valid_pairs).fill_(0)
        concordant_pairs = torch.logical_or(
            (y_diff < 0) & (y_hat_diff < 0) & (status_i == 1),
            (y_diff > 0) & (y_hat_diff > 0) & (status_j == 1)
        ).float()
        torch.diagonal(concordant_pairs).fill_(0)
        c_index = concordant_pairs.sum() / (valid_pairs.sum() + 1e-6)
        return c_index.item()

###############################################################################
# 5. Adversarial G/D
###############################################################################
from scipy.stats import genpareto
genpareto_params = (0.5, 0, 1.0)
threshold = 2.0
rv = genpareto(*genpareto_params)

def sample_extreme_code(batch_size, extreme_dim=1):
    probs = torch.rand(batch_size, extreme_dim) * 0.95
    samples = rv.ppf(probs.numpy()) + threshold
    return torch.tensor(samples, dtype=torch.float32)

class TabularGeneratorWithRealInput(nn.Module):
    def __init__(self, in_dim, latent_dim=10, extreme_dim=1, hidden_dim=64):
        super(TabularGeneratorWithRealInput, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + latent_dim + extreme_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
    def forward(self, real_features, noise, extreme_code):
        x = torch.cat((real_features, noise, extreme_code), dim=1)
        return self.net(x)

class TabularDiscriminator(nn.Module):
    def __init__(self, in_dim=20, extreme_dim=1, hidden_dim=64):
        super(TabularDiscriminator, self).__init__()
        self.fc1 = nn.Linear(in_dim + extreme_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, features, extreme_code):
        x = torch.cat((features, extreme_code), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

###############################################################################
# 6. Example IG function & penalty
###############################################################################
def compute_ig_for_interval(model, baseline, input_sample, interval_idx, steps=20):
    # a simplified version using torch.autograd.grad
    device = input_sample.device
    batch_size, in_dim = input_sample.shape

    alphas = torch.linspace(0, 1, steps).to(device)
    baseline_exp = baseline.unsqueeze(0).expand(steps, batch_size, in_dim)
    input_exp    = input_sample.unsqueeze(0).expand(steps, batch_size, in_dim)
    interpolated = baseline_exp + alphas.view(-1,1,1)*(input_exp - baseline_exp)

    total_grad = torch.zeros_like(interpolated, device=device)
    for s in range(steps):
        x_s = interpolated[s].clone().detach().requires_grad_(True)
        out_s = model(x_s)[interval_idx]
        out_sum = out_s.sum()
        grad_s = torch.autograd.grad(
            out_sum, 
            x_s, 
            retain_graph=False,
            create_graph=False
        )[0]
        total_grad[s] = grad_s

    avg_grad = total_grad.mean(dim=0)  
    ig = (input_sample - baseline) * avg_grad
    return ig

def ig_smoothness_penalty(ig_list):
    penalty = 0.0
    for j in range(len(ig_list)-1):
        diff = torch.abs(ig_list[j+1] - ig_list[j])
        penalty += diff.sum()
    return penalty

import matplotlib.pyplot as plt

###############################################################################
# 7.1 - Expert attribution penalty
###############################################################################
def expert_attribution_penalty(
    ig_all, 
    expert_features, 
    lambda_1, 
    lambda_2
):
    """
    ig_all: Tensor, shape=(batch_size, num_tasks, num_features)
            对每个样本、每个区间(task)、每个特征(feature)的IG。
    expert_features: list/set, 表示医生标记为“重要”的特征index。
    lambda_1, lambda_2: 惩罚系数(超参), 分别对应公式两项。

    计算:
      Omega_expert(\Phi)
       = lambda_1 * sum_{l' in I} ReLU( average - ||Phi_{l'}||_2 )
         + lambda_2 * sum_{l'' not in I} ||Phi_{l''}||_2

    其中 ||Phi_l||_2 表示特征 l 在(batch_size * num_tasks)维度上的 L2 范数，
    average = (1/p) * sum_{l=1..p} ||Phi_l||_2.
    """
    B, T, P = ig_all.shape  # B=batch_size, T=num_tasks, P=num_features

    # reshape -> (B*T, P)
    ig_reshape = ig_all.view(-1, P)  # 对所有样本、所有任务拼接

    # 计算每个特征在本 batch+任务下的 L2 范数
    norms = []
    for l in range(P):
        norm_l = torch.norm(ig_reshape[:, l], p=2)
        norms.append(norm_l)
    norms = torch.stack(norms)  # shape=(P,)

    avg_norm = norms.mean()
    expert_set = set(expert_features)
    not_expert = [l for l in range(P) if l not in expert_set]

    # 重要特征：ReLU( average - ||Phi_l'||_2 )
    loss_exp = 0.0
    for l_prime in expert_set:
        diff_val = F.relu(avg_norm - norms[l_prime])
        loss_exp += diff_val

    # 不重要特征：||Phi_{l''}||_2
    loss_nonexp = 0.0
    for l_prime in not_expert:
        loss_nonexp += norms[l_prime]

    penalty = lambda_1 * loss_exp + lambda_2 * loss_nonexp
    return penalty


###############################################################################
# 7.3 - The combined Trainer with Expert Knowledge
###############################################################################
class EGTrainerAttribution:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        train_dataset,
        args,
        generator,
        discriminator,
        expert_features
    ):
        """
        主要改动:
        - 加入 expert_features: 表示医生先验的“重要”特征索引集合
        - 去掉时间平滑惩罚, 仅保留专家先验正则
        - 保留对抗训练 (GAN) 部分, 若不需要可自行删除
        """
        self.device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.train_dataset = train_dataset
        self.epochs = args.epochs
        self.clip = args.clip
        self.args = args

        # 优化器 & 学习率调度
        self.optimizer_model = optim.NAdam(self.model.parameters(), lr=args.lr, weight_decay=1e-2)
        self.scheduler = ReduceLROnPlateau(self.optimizer_model, mode='min', factor=0.1, patience=5)
        
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop_patience = args.early_stop_patience
        self.checkpoint_path = os.path.join(args.save_path, "best_model.pth")

        # 对抗网络
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.optG = optim.Adam(self.generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
        self.optD = optim.Adam(self.discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
        self.bce = nn.BCELoss()

        # 专家先验特征
        self.expert_features = expert_features
        # 两个正则系数(在 args 中声明, 需 args.lambda_1, args.lambda_2)
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2

        # IG 计算步数
        self.steps_ig = args.steps_ig

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, 10).to(self.device)

    def train(self):
        from scipy.stats import genpareto
        rv = genpareto(0.5, 0, 1.0)
        threshold = 2.0

        for epoch in trange(self.epochs):
            self.model.train()
            running_loss = 0.0

            for X_batch, targets, masks, event in self.train_loader:
                X_batch = X_batch.to(self.device)
                targets = [t.to(self.device) for t in targets]
                masks   = [m.to(self.device) for m in masks]
                batch_size = X_batch.size(0)

                #############################################
                # 1) Adversarial training
                #############################################
                # 采样极值扰动
                probs = torch.rand(batch_size, 1).to(self.device) * 0.95
                extreme_samples = rv.ppf(probs.cpu().numpy()) + threshold
                extreme_code = torch.tensor(extreme_samples, dtype=torch.float32).to(self.device)

                noise = self.sample_noise(batch_size)
                X_fake = self.generator(X_batch, noise, extreme_code)

                # (a) Discriminator
                self.optD.zero_grad()
                d_real = self.discriminator(X_batch, torch.zeros(batch_size, 1, device=self.device))
                r_label = torch.ones(batch_size, 1, device=self.device)
                d_loss_real = self.bce(d_real, r_label)

                d_fake = self.discriminator(X_fake.detach(), extreme_code)
                f_label = torch.zeros(batch_size, 1, device=self.device)
                d_loss_fake = self.bce(d_fake, f_label)
                (d_loss_real + d_loss_fake).backward()
                self.optD.step()

                # (b) Generator
                self.optG.zero_grad()
                d_fake_g = self.discriminator(X_fake, extreme_code)
                g_loss = self.bce(d_fake_g, r_label)
                g_loss.backward()
                self.optG.step()

                #############################################
                # 2) 计算 IG + 专家先验正则
                #############################################
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad_(False)

                X_fake_detached = X_fake.detach()
                baseline = X_fake_detached                

                ig_per_interval = []
                for j in range(self.model.num_tasks):
                    IG_j = compute_ig_for_interval(
                        self.model,
                        baseline,
                        X_batch,
                        interval_idx=j,
                        steps=self.steps_ig
                    )
                    ig_per_interval.append(IG_j.detach())

                for param in self.model.parameters():
                    param.requires_grad_(True)
                self.model.train()

                ig_tensor = torch.stack(ig_per_interval, dim=1)  # (batch_size, num_tasks, num_features)

                pen_expert = expert_attribution_penalty(
                    ig_all=ig_tensor,
                    expert_features=self.expert_features,
                    lambda_1=self.lambda_1,
                    lambda_2=self.lambda_2
                )

                #############################################
                # 3) survival loss + expert penalty
                #############################################
                outputs = self.model(X_batch)
                surv_loss = self.model.custom_loss(outputs, targets, masks)
                total_loss = surv_loss + pen_expert

                self.optimizer_model.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer_model.step()

                running_loss += total_loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

            stop_early = self.validate(epoch)
            if stop_early:
                break

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, val_targets, val_masks, event_val in self.test_loader:
                X_val = X_val.to(self.device)
                val_targets = [vt.to(self.device) for vt in val_targets]
                val_masks   = [vm.to(self.device) for vm in val_masks]
                val_outputs = self.model(X_val)
                loss = self.model.custom_loss(val_outputs, val_targets, val_masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.test_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        current_lr = self.optimizer_model.param_groups[0]['lr']
        self.scheduler.step(avg_val_loss)
        print(f"Current LR: {current_lr:.6f}")

        is_best = (avg_val_loss < self.best_val_loss)
        if is_best:
            self.best_val_loss = avg_val_loss
            self.epochs_no_improve = 0
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_val_loss': self.best_val_loss,
                'optimizer': self.optimizer_model.state_dict(),
            }, is_best=True)
        else:
            self.epochs_no_improve += 1

        if (self.epochs_no_improve >= self.early_stop_patience) or (current_lr <= 1e-8):
            print(f'Early stopping triggered at epoch {epoch+1}.')
            return True
        return False

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        torch.save(state, os.path.join(self.args.save_path, filename))
        if is_best:
            torch.save(state, os.path.join(self.args.save_path, "best_model.pth"))

    def load_best_checkpoint(self):
        ckpt_path = os.path.join(self.args.save_path, "best_model.pth")
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer_model.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_val_loss = ckpt['best_val_loss']
        print(f"Loaded from {ckpt_path}, epoch={start_epoch}, best_val_loss={best_val_loss:.4f}")

    def predict(self, data_loader):
        """
        返回:
          - predictions: shape=(N, num_tasks)
          - Y_hat:       shape=(N,), 每个样本存活区间数(二值化后求和)
          - Y_true:      shape=(N,), 真实的存活区间数
          - events:      shape=(N,), 事件指标(0或1)
        """
        self.model.eval()
        predictions = []
        Y_true = []
        events = []
        idx_offset = 0

        with torch.no_grad():
            for i, (X, targets, masks, status) in enumerate(data_loader):
                X = X.to(self.device)
                outs = self.model(X)  # list of [batch_size, 1] for each task
                bsz = X.shape[0]
                events.append(status.numpy())

                out_block = torch.cat(outs, dim=1)   # shape=(bsz, num_tasks)
                predictions.append(out_block.cpu().numpy())

                true_block = torch.cat(targets, dim=1)  # shape=(bsz, num_tasks)
                Y_true.append(true_block.cpu().numpy())
                idx_offset += bsz

        predictions = np.concatenate(predictions, axis=0)  # (N, num_tasks)
        Y_true      = np.concatenate(Y_true, axis=0)       # (N, num_tasks)
        events      = np.concatenate(events, axis=0)       # (N,)

        Y_hat = (predictions > 0.5).astype(int).sum(axis=1)  # 每行统计存活区间数
        Y_sum = Y_true.sum(axis=1)                           # 真值存活区间数

        return predictions, Y_hat, Y_sum, events

    ###########################################################################
    # 可视化特征重要性 (Global & Local)
    ###########################################################################
    def compute_global_feature_importance(self, data_loader, baseline_type="zero"):
        """
        在 data_loader 上计算全局特征重要性:
          1) 对所有样本 & 任务计算 IG
          2) 在 (N, T) 上累加或取均值
          3) 得到每个特征的平均贡献(带正负)
        baseline_type: 选择基线; 这里默认全 0
        返回: mean_importance, shape=(num_features,)
        """
        self.model.eval()
        importance_accum = None

        for X_batch, targets, masks, event in data_loader:
            X_batch = X_batch.to(self.device)
            batch_size = X_batch.size(0)

            if baseline_type == "zero":
                baseline = torch.zeros_like(X_batch)
            else:
                # 可自行实现其他 baseline, 如: baseline = X_batch.min() / ...
                baseline = torch.zeros_like(X_batch)

            ig_total = torch.zeros_like(X_batch)
            for j in range(self.model.num_tasks):
                IG_j = compute_ig_for_interval(
                    self.model, baseline, X_batch, interval_idx=j, steps=self.steps_ig
                )
                ig_total += IG_j  # 把不同task的IG累加

            ig_np = ig_total.detach().cpu().numpy()
            if importance_accum is None:
                importance_accum = ig_np
            else:
                importance_accum = np.vstack([importance_accum, ig_np])

        mean_importance = importance_accum.mean(axis=0)  # shape=(num_features,)
        return mean_importance

    def compute_local_feature_importance(self, X_sample, baseline_type="zero"):
        """
        对单个样本计算局部特征重要性(累加所有任务).
        返回: local_importance, shape=(num_features,).
        """
        self.model.eval()
        X_sample = X_sample.unsqueeze(0).to(self.device)  # (1, num_features)
        if baseline_type == "zero":
            baseline = torch.zeros_like(X_sample)
        else:
            baseline = torch.zeros_like(X_sample)

        ig_total = torch.zeros_like(X_sample)
        for j in range(self.model.num_tasks):
            IG_j = compute_ig_for_interval(
                self.model, baseline, X_sample, interval_idx=j, steps=self.steps_ig
            )
            ig_total += IG_j

        return ig_total.squeeze(0).cpu().numpy()

    def plot_top_k_features(self, importance, feature_names, k=10, title="Feature Importances"):
        """
        不带 save_file 参数。根据 title 中包含的关键字:
        - 'entire':  保存为 top_k_features_entire.png
        - 'train':   保存为 top_k_features_train.png
        - 'test':    保存为 top_k_features_test.png
        - 'local':   保存为 top_k_features_local.png
        否则统一保存 top_k_features.png
        """
        import numpy as np
        import matplotlib.pyplot as plt

        idx_sorted = np.argsort(np.abs(importance))[::-1]
        top_idx = idx_sorted[:k]

        top_features = [feature_names[i] for i in top_idx]
        top_scores   = importance[top_idx]

        top_pairs = sorted(zip(top_features, top_scores), key=lambda x: abs(x[1]))
        feat_sorted = [x[0] for x in top_pairs]
        vals_sorted = [x[1] for x in top_pairs]

        plt.figure(figsize=(8, 6))
        bar_positions = np.arange(len(feat_sorted))
        colors = ["royalblue" if v >= 0 else "tomato" for v in vals_sorted]
        plt.barh(bar_positions, vals_sorted, color=colors)
        plt.yticks(bar_positions, feat_sorted)
        plt.title(title)
        plt.xlabel("IG contribution")
        plt.axvline(x=0, color='black', linewidth=0.8)
        plt.tight_layout()

        # 根据标题自动选定文件名
        t_lower = title.lower()
        if "entire" in t_lower:
            filename = "top_k_features_entire.png"
        elif "train" in t_lower:
            filename = "top_k_features_train.png"
        elif "test" in t_lower:
            filename = "top_k_features_test.png"
        elif "local" in t_lower:
            filename = "top_k_features_local.png"
        else:
            filename = "top_k_features.png"

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

###############################################################################
# 2) Main
###############################################################################
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # (A) 数据加载 & 预处理  (您已在前面做了, 此处仅示例)
    # -------------------------------------------------------------------------
    #   1) X_train, X_test
    #   2) Y_train_transform, Y_test_transform
    #   3) W_train_transform, W_test_transform
    #   4) train_loader, test_loader
    #   5) event_train, event_test
    #   6) X_columns = X.columns
    #
    # 假设这些变量都已准备好, 并且 X_train.shape[1] 是 num_features
    # -------------------------------------------------------------------------

    # -----------------------
    # (B) 从 Excel 读入重要基因
    # -----------------------
    import pandas as pd

    excel_path = "/project/lwang/xqin5/SAcompare/TCGA_Task3/Importancegenes.xlsx"
    df_genes = pd.read_excel(excel_path)
    # 第一列存放基因名字, 前 474 行是感兴趣的基因
    # 先取出前 474 个基因名:
    interest_names = df_genes.iloc[:474, 0].tolist()  # 将前474行的第一列转为list

    # X_columns: 特征矩阵的列名
    # 我们需要找到 X_columns 当中与 interest_names 相匹配的索引
    X_columns_list = list(X_columns)  # 若 X_columns 本身是Index, 这里转成list

    # 生成 expert_features (存放这些基因在 X_columns 中的下标)
    expert_features = []
    for i, col in enumerate(X_columns_list):
        if col in interest_names:
            expert_features.append(i)

    # 打印一下匹配到的数量, 以确保找到的索引数是 <= 474
    print(f"Number of matched expert features: {len(expert_features)}")

    # (C) 构建模型/生成器/判别器
    print("\n=== Building model/generator/discriminator ===")
    multi_task_model = MultiTaskModel(
        in_features=X_train.shape[1],
        out_features=Y_train.shape[1],
        hidden_dim=128
    )
    generator = TabularGeneratorWithRealInput(
        in_dim=X_train.shape[1],
        latent_dim=10,
        extreme_dim=1,
        hidden_dim=64
    )
    discriminator = TabularDiscriminator(
        in_dim=X_train.shape[1],
        extreme_dim=1,
        hidden_dim=64
    )

    # (D) 准备 Trainer (带专家先验特征)
    print("\n=== Instantiate Trainer with Expert penalty ===")
    trainer = EGTrainerAttribution(
        model=multi_task_model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        args=args,  # 需保证 args 内含: lambda_1, lambda_2, steps_ig, ...
        generator=generator,
        discriminator=discriminator,
        expert_features=expert_features  # 这次不再随机，而是基于Excel匹配
    )

    # (E) 训练
    print("\n=== Start training ===")
    trainer.train()

    # (F) 加载最佳模型
    print("\n=== Load best checkpoint ===")
    trainer.load_best_checkpoint()

    # (G) 评估 C-index
    print("\n=== Evaluate on training set ===")
    predictions, Y_hat, Y_true, ev = trainer.predict(train_loader)
    cindex_fn = EditedCindexOptimized()
    c_train = cindex_fn(Y_true, Y_hat, ev)
    print(f"C-index (train): {c_train:.4f}")

    print("\n=== Evaluate on test set ===")
    predictions, Y_hat, Y_true, ev = trainer.predict(test_loader)
    c_test = cindex_fn(Y_true, Y_hat, ev)
    print(f"C-index (test): {c_test:.4f}")

    # (H) 分别画 Entire / Train / Test / Local
    # 先合并 entire_loader (train + test), 以绘制 "Entire"
    X_all = torch.cat([X_train, X_test], dim=0)
    Y_all_2d = torch.cat([torch.cat(Y_train_transform, dim=1), torch.cat(Y_test_transform, dim=1)], dim=0)
    W_all_2d = torch.cat([torch.cat(W_train_transform, dim=1), torch.cat(W_test_transform, dim=1)], dim=0)
    event_all = np.concatenate([event_train, event_test], axis=0)

    Y_all_transform = [Y_all_2d[:, i:i+1] for i in range(Y_all_2d.size(1))]
    W_all_transform = [W_all_2d[:, i:i+1] for i in range(W_all_2d.size(1))]

    entire_dataset = MultiTaskDataset(X_all, Y_all_transform, W_all_transform, event_all)
    entire_loader = DataLoader(entire_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print("\n=== Global Feature Importance on ENTIRE dataset ===")
    global_importance_entire = trainer.compute_global_feature_importance(entire_loader, baseline_type="zero")
    trainer.plot_top_k_features(
        importance=global_importance_entire,
        feature_names=X_columns_list,
        k=10,
        title="Global Feature Importance (Entire dataset)"
    )

    print("\n=== Global Feature Importance on TRAIN set ===")
    global_importance_train = trainer.compute_global_feature_importance(train_loader, baseline_type="zero")
    trainer.plot_top_k_features(
        importance=global_importance_train,
        feature_names=X_columns_list,
        k=10,
        title="Global Feature Importance (Train set)"
    )

    print("\n=== Global Feature Importance on TEST set ===")
    global_importance_test = trainer.compute_global_feature_importance(test_loader, baseline_type="zero")
    trainer.plot_top_k_features(
        importance=global_importance_test,
        feature_names=X_columns_list,
        k=10,
        title="Global Feature Importance (Test set)"
    )

    # Local特征重要性示例 (取测试集中第一个样本)
    print("\n=== Local Feature Importance on ONE sample ===")
    X_one_sample = X_test[0]
    local_importance = trainer.compute_local_feature_importance(X_one_sample, baseline_type="zero")
    trainer.plot_top_k_features(
        importance=local_importance,
        feature_names=X_columns_list,
        k=10,
        title="Local Feature Importance (Sample #0)"
    )

    print("Done.")
