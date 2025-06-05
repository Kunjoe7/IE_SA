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
df = pd.read_csv('/project/lwang/xqin5/SAcompare/NWTCO/nwtco.csv')
data = df.copy()
data = data.drop(data[data['edrel'] <= 0].index)
data.dropna(inplace=True)
print("Is there any NaN?", (data.isna().sum() > 0).any())
print("DATA SHAPE: ", data.shape)

data_original = data.copy(deep=True)

args = easydict.EasyDict({
    "batch_size": 64,
    "cuda": True,
    "lr": 0.005,        
    "epochs": 200,
    "clip": 5.0,
    "lambda_reg": 0.01,
    "save_path": "outputfiles",
    "eg_k": 1,
    "early_stop_patience": 11,
    "lambda_smooth": 0.01,   
    "steps_ig": 20,             "g_lr": 1e-4,           
    "d_lr": 1e-4,           
})

print(f"Survival time min: {data.edrel.min()}, max: {data.edrel.max()}")

###############################################################################
# 2.1 Create intervals and label/mask vectors
###############################################################################
data_sorted = data.sort_values(by='edrel').reset_index(drop=True)
bin_size = 150
data_sorted['interval_number'] = data_sorted.index // bin_size + 1
bins_distribution = data_sorted['interval_number'].value_counts().sort_index()
num_intervals = len(data_sorted['interval_number'].unique())
print("Num Intervals: ", num_intervals)
print(bins_distribution)

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
        return mv.tolist()
    else:
        mv = np.zeros(num_intervals)
        mv[:interval_number] = 1
        return mv.tolist()

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

###############################################################################
# 7. The combined Trainer
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
        discriminator
    ):
        self.device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.train_dataset = train_dataset
        self.epochs = args.epochs
        self.clip = args.clip
        self.args = args

        self.optimizer_model = optim.NAdam(self.model.parameters(), lr=args.lr, weight_decay=1e-2)
        # Fix: store "self.optimizer_model" -> pass it to scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer_model, mode='min', factor=0.1, patience=5)
        
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop_patience = args.early_stop_patience
        self.checkpoint_path = os.path.join(args.save_path, "best_model.pth")

        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.optG = optim.Adam(self.generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
        self.optD = optim.Adam(self.discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
        self.bce = nn.BCELoss()

        self.lambda_smooth = args.lambda_smooth
        self.steps_ig = args.steps_ig

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, 10).to(self.device)

    def train(self):
        for epoch in trange(self.epochs):
            self.model.train()
            running_loss = 0.0

            for X_batch, targets, masks, event in self.train_loader:
                X_batch = X_batch.to(self.device)
                targets = [t.to(self.device) for t in targets]
                masks   = [m.to(self.device) for m in masks]
                batch_size = X_batch.size(0)

                # 1) Adversarial training
                extreme_code = sample_extreme_code(batch_size, 1).to(self.device)
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

                # detach
                X_fake_detached = X_fake.detach()

                # 2) IG
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad_(False)

                ig_per_interval = []
                for j in range(self.model.num_tasks):
                    IG_j = compute_ig_for_interval(
                        self.model,
                        X_fake_detached,
                        X_batch,
                        interval_idx=j,
                        steps=self.steps_ig
                    )
                    ig_per_interval.append(IG_j.detach())

                # re-enable
                for param in self.model.parameters():
                    param.requires_grad_(True)
                self.model.train()

                pen_smooth = ig_smoothness_penalty(ig_per_interval)

                # 3) survival loss
                outputs = self.model(X_batch)
                surv_loss = self.model.custom_loss(outputs, targets, masks)
                total_loss = surv_loss + self.lambda_smooth * pen_smooth

                self.optimizer_model.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer_model.step()

                running_loss += total_loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}")
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
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}")

        # Get LR from self.optimizer_model param groups
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
        self.model.eval()
        predictions = np.zeros((len(data_loader)*self.args.batch_size, self.model.num_tasks))
        Y_true = np.zeros((len(data_loader)*self.args.batch_size, self.model.num_tasks))
        events = np.zeros((len(data_loader)*self.args.batch_size))
        idx_offset = 0

        with torch.no_grad():
            for i, (X, targets, masks, status) in enumerate(data_loader):
                X = X.to(self.device)
                outs = self.model(X)
                bsz = X.shape[0]
                events[idx_offset: idx_offset+bsz] = status.cpu().numpy()

                for j, out_j in enumerate(outs):
                    predictions[idx_offset: idx_offset+bsz, j] = out_j.cpu().numpy()[:, 0]
                    Y_true[idx_offset: idx_offset+bsz, j]      = targets[j].cpu().numpy()[:, 0]
                idx_offset += bsz

        Y_true = np.sum(Y_true[:idx_offset], axis=1)
        Y_hat = (predictions[:idx_offset] > 0.5).astype(int)
        Y_hat = np.sum(Y_hat, axis=1)
        return predictions[:idx_offset], Y_hat, Y_true, events[:idx_offset]


###############################################################################
# 8. Main
###############################################################################
if __name__ == "__main__":
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

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

    print("\n=== Instantiate Trainer ===")
    trainer = EGTrainerAttribution(
        model=multi_task_model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        args=args,
        generator=generator,
        discriminator=discriminator
    )

    print("\n=== Start training ===")
    trainer.train()

    print("\n=== Load best checkpoint ===")
    trainer.load_best_checkpoint()

    print("\n=== Evaluate on training set ===")
    predictions, Y_hat, Y_true, ev = trainer.predict(train_loader)
    cindex_fn = EditedCindexOptimized()
    c_train   = cindex_fn(Y_true, Y_hat, ev)
    print(f"C-index (train): {c_train:.4f}")

    print("\n=== Evaluate on test set ===")
    predictions, Y_hat, Y_true, ev = trainer.predict(test_loader)
    c_test = cindex_fn(Y_true, Y_hat, ev)
    print(f"C-index (test): {c_test:.4f}")
    print("Done.")
