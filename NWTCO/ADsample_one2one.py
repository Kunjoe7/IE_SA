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
# 2. Load and preprocess your original data (NWTCO dataset etc.)
###############################################################################
df = pd.read_csv('/project/lwang/xqin5/SAcompare/NWTCO/nwtco.csv')
data = df.copy()
# Drop rows where 'edrel' <= 0 or NaN
data = data.drop(data[data['edrel'] <= 0].index)
data.dropna(inplace=True)
print("Is there any NaN?", (data.isna().sum() > 0).any())
print("DATA SHAPE: ", data.shape)

data_original = data.copy(deep=True)
args = easydict.EasyDict({
    "batch_size": 64,
    "cuda": True, 
    "lr": 0.01,
    "epochs": 200,
    "clip": 5.0,
    "lambda_reg": 0.01,
    "save_path": "outputfiles",
    "eg_k" : 1,
    "early_stop_patience": 11,
})

print(f"Survival time min: {data.edrel.min()}, max: {data.edrel.max()}")

###############################################################################
# 2.1 Create intervals and label/mask vectors as in your original pipeline
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
# 2.2 Define MultiTaskDataset
###############################################################################
class MultiTaskDataset(Dataset):
    def __init__(self, X, Y_list, M_list, events):
        """
        X: (N, d)
        Y_list: list of T tensors, each (N, 1)
        M_list: list of T tensors, each (N, 1)
        events: (N,)
        """
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
# 3. Define the MultiTaskModel (SimplifiedSharedLayer) as in your original code
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

class MultiTaskModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=128):
        super(MultiTaskModel, self).__init__()
        self.num_tasks = out_features
        self.shared_layer = SimplifiedSharedLayer(in_features, hidden_dim)
        self.task_layers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(out_features)])

    def forward(self, x):
        shared_output = self.shared_layer(x)
        task_outputs = []
        for i, task_layer in enumerate(self.task_layers):
            if i == 0:
                task_output = torch.sigmoid(task_layer(shared_output))
            else:
                task_output = torch.sigmoid(task_layer(shared_output)) * task_outputs[-1]
            task_outputs.append(task_output)
        return task_outputs

    def custom_loss(self, task_outputs, targets, masks):
        loss = 0
        for i, task_output in enumerate(task_outputs):
            task_target = targets[i]
            task_mask = masks[i]
            task_loss = F.binary_cross_entropy(task_output, task_target.float(), reduction='none')
            task_loss = task_loss * task_mask.float()
            loss += task_loss.sum() / (task_mask.sum() + 1e-6)
        return loss

###############################################################################
# 4. A simplified c-index calculator
###############################################################################
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
        y_diff = y.unsqueeze(1) - y.unsqueeze(0)
        y_hat_diff = y_hat.unsqueeze(1) - y_hat.unsqueeze(0)
        status_i = status.unsqueeze(1)
        status_j = status.unsqueeze(0)
        valid_pairs = torch.logical_or((y_diff < 0) & (status_i == 1), (y_diff > 0) & (status_j == 1)).float()
        torch.diagonal(valid_pairs).fill_(0)
        concordant_pairs = torch.logical_or(
            (y_diff < 0) & (y_hat_diff < 0) & (status_i == 1),
            (y_diff > 0) & (y_hat_diff > 0) & (status_j == 1)
        ).float()
        torch.diagonal(concordant_pairs).fill_(0)
        c_index = concordant_pairs.sum() / (valid_pairs.sum() + 1e-6)
        return c_index.item()

###############################################################################
# 5. Your EGTrainer (or NoRegularizationTrainer) for multi-task model
###############################################################################
from torch.optim import NAdam

class EGTrainer:
    def __init__(self, model, train_loader, test_loader, train_dataset, args):
        self.device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
        self.model = model.to(self.device)
        self.optimizer = NAdam(self.model.parameters(), lr=args.lr, weight_decay=1e-2)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_dataset = train_dataset
        self.epochs = args.epochs
        self.clip = args.clip
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop_patience = args.early_stop_patience
        self.checkpoint_path = os.path.join(args.save_path, "best_model.pth")
        self.args = args

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        torch.save(state, os.path.join(self.args.save_path, filename))
        if is_best:
            torch.save(state, os.path.join(self.args.save_path, "best_model.pth"))

    def train(self):
        for epoch in trange(self.epochs):
            self.model.train()
            running_loss = 0.0
            for batch_idx, (X_train, targets, masks, event_train) in enumerate(self.train_loader):
                X_train = X_train.to(self.device)
                X_train.requires_grad = True
                targets = [target.to(self.device) for target in targets]
                masks = [mask.to(self.device) for mask in masks]
                task_outputs = self.model(X_train)
                loss = self.model.custom_loss(task_outputs, targets, masks)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(self.train_loader)
            print(f'Epoch {epoch}, Training Loss: {avg_loss:.4f}')
            early_stop = self.validate(epoch)
            if early_stop:
                return

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, val_targets, val_masks, event_val in self.test_loader:
                X_val = X_val.to(self.device)
                val_targets = [target.to(self.device) for target in val_targets]
                val_masks = [mask.to(self.device) for mask in val_masks]
                val_outputs = self.model(X_val)
                loss = self.model.custom_loss(val_outputs, val_targets, val_masks)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(self.test_loader)
        print(f'End of Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}')
        self.scheduler.step(avg_val_loss)
        current_lr = self.scheduler.optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr:.6f}')
        is_best = avg_val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = avg_val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        self.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'optimizer': self.optimizer.state_dict(),
        }, is_best)
        if (self.epochs_no_improve >= self.early_stop_patience) or (current_lr == 0.0):
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            return True
        return False

    def load_best_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.args.save_path, "best_model.pth"))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded model from checkpoint at epoch {start_epoch} with best validation loss {best_val_loss:.4f}")

    def predict(self, data_loader):
        self.model.eval()
        predictions = np.zeros((len(data_loader)*self.args.batch_size, self.model.num_tasks))
        Y_true = np.zeros((len(data_loader)*self.args.batch_size, self.model.num_tasks))
        events = np.zeros((len(data_loader)*self.args.batch_size))
        idx_offset = 0
        with torch.no_grad():
            for i, (X, targets, masks, status) in enumerate(data_loader):
                X = X.to(self.device)
                task_outputs_ = self.model(X)
                batch_len = X.shape[0]
                events[idx_offset: idx_offset+batch_len] = status.cpu().numpy()
                for j, task_output in enumerate(task_outputs_):
                    predictions[idx_offset: idx_offset+batch_len, j] = task_output.cpu().numpy()[:, 0]
                    Y_true[idx_offset: idx_offset+batch_len, j] = targets[j].cpu().numpy()[:, 0]
                idx_offset += batch_len
        Y_true = np.sum(Y_true[:idx_offset], axis=1)
        Y_hat = (predictions[:idx_offset] > 0.5).astype(int)
        Y_hat = np.sum(Y_hat, axis=1)
        return predictions[:idx_offset], Y_hat, Y_true, events[:idx_offset]


###############################################################################
# 6. One-to-one Generator: Takes (RealFeatures + Noise + ExtremeCode) -> Fake
###############################################################################
from scipy.stats import genpareto

# Example genpareto params (shape, loc, scale). Adjust as needed.
genpareto_params = (0.5, 0, 1.0)
threshold = 2.0
rv = genpareto(*genpareto_params)

def sample_extreme_code(batch_size, extreme_dim=1):
    """
    Sample from a generalized Pareto distribution, then add a threshold to push the values higher.
    """
    probs = torch.rand(batch_size, extreme_dim) * 0.95
    samples = rv.ppf(probs.numpy()) + threshold
    return torch.tensor(samples, dtype=torch.float32)

class TabularGeneratorWithRealInput(nn.Module):
    """
    This generator takes real features, noise, and an extreme code to generate a 'one-to-one'
    adversarial sample that is a perturbed version of the real sample with extreme properties.
    """
    def __init__(self, in_dim, latent_dim=10, extreme_dim=1, out_dim=20, hidden_dim=64):
        super(TabularGeneratorWithRealInput, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + latent_dim + extreme_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)  # out_dim must match the real feature dimension
        )

    def forward(self, real_features, noise, extreme_code):
        # real_features shape: (batch_size, in_dim)
        # noise shape: (batch_size, latent_dim)
        # extreme_code shape: (batch_size, extreme_dim)
        x = torch.cat((real_features, noise, extreme_code), dim=1)
        out = self.net(x)
        return out

###############################################################################
# 7. One-to-one adversarial training
###############################################################################
class ExtremeAdversarialTrainerOneToOne:
    """
    For each real sample in the loader, generate a corresponding adversarial sample
    using the generator (real_features + noise + extreme_code).
    """
    def __init__(
        self,
        generator,
        discriminator,
        real_loader,
        in_dim=20,
        latent_dim=10,
        extreme_dim=1,
        g_lr=1e-4,
        d_lr=1e-4,
        device="cpu"
    ):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.real_loader = real_loader
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.extreme_dim = extreme_dim
        self.optG = optim.Adam(self.G.parameters(), lr=g_lr, betas=(0.5, 0.999))
        self.optD = optim.Adam(self.D.parameters(), lr=d_lr, betas=(0.5, 0.999))
        self.bce = nn.BCELoss()
        self.device = device

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.latent_dim).to(self.device)

    def train(self, epochs=5):
        """
        Train for a certain number of epochs. Each epoch goes through
        all real data, generating one-to-one fake samples, then updating
        discriminator and generator.
        """
        for epoch in range(epochs):
            total_d_loss = 0
            total_g_loss = 0
            for (X_real, Y, M, event) in self.real_loader:
                batch_size = X_real.size(0)
                X_real = X_real.to(self.device)
                # sample extreme code
                extreme_code = sample_extreme_code(batch_size, self.extreme_dim).to(self.device)
                # sample random noise
                noise = self.sample_noise(batch_size)
                # Generate fake sample => G(real_features + noise + extreme_code)
                X_fake = self.G(X_real, noise, extreme_code)
                
                # Train discriminator
                self.optD.zero_grad()
                # For real data, we can feed (X_real, maybe zeros as code) or simply skip code
                # Here let's put code=0 for real data to separate from extreme
                d_real = self.D(X_real, torch.zeros(batch_size, self.extreme_dim).to(self.device))
                d_fake = self.D(X_fake.detach(), extreme_code)

                label_real = torch.ones(batch_size, 1).to(self.device)
                label_fake = torch.zeros(batch_size, 1).to(self.device)
                d_loss_real = self.bce(d_real, label_real)
                d_loss_fake = self.bce(d_fake, label_fake)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optD.step()

                # Train generator
                self.optG.zero_grad()
                d_fake_g = self.D(X_fake, extreme_code)
                g_adv_loss = self.bce(d_fake_g, label_real)
                g_loss = g_adv_loss
                g_loss.backward()
                self.optG.step()

                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()

            avg_d_loss = total_d_loss / len(self.real_loader)
            avg_g_loss = total_g_loss / len(self.real_loader)
            print(f"[Epoch {epoch+1}/{epochs}] D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")

    def generate_one_to_one_samples(self):
        """
        Generate an adversarial sample for each real sample in the loader, one by one.
        Return a list or array of shape (N, in_dim) for the fake samples, plus the codes.
        """
        all_real = []
        all_fake = []
        all_codes = []
        self.G.eval()
        with torch.no_grad():
            for (X_real, Y, M, event) in self.real_loader:
                batch_size = X_real.size(0)
                X_real = X_real.to(self.device)
                extreme_code = sample_extreme_code(batch_size, self.extreme_dim).to(self.device)
                noise = self.sample_noise(batch_size)
                X_fake = self.G(X_real, noise, extreme_code)
                all_real.append(X_real.cpu().numpy())
                all_fake.append(X_fake.cpu().numpy())
                all_codes.append(extreme_code.cpu().numpy())

        all_real = np.concatenate(all_real, axis=0)
        all_fake = np.concatenate(all_fake, axis=0)
        all_codes = np.concatenate(all_codes, axis=0)
        return all_real, all_fake, all_codes

###############################################################################
# 8. Define a simple Discriminator for the one-to-one approach
###############################################################################
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
# 9. Putting everything together:
#    - Train the one-to-one adversarial model
#    - Generate and save adversarial samples for each real sample
#    - Train or evaluate your multi-task model as usual
###############################################################################
if __name__ == "__main__":
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

    # 1) Instantiate the one-to-one generator and discriminator
    in_dim = X_train.shape[1]  # same as out_dim
    latent_dim = 10
    extreme_dim = 1
    hidden_dim = 64
    generator_one2one = TabularGeneratorWithRealInput(
        in_dim=in_dim,
        latent_dim=latent_dim,
        extreme_dim=extreme_dim,
        out_dim=in_dim,
        hidden_dim=hidden_dim
    )
    discriminator_one2one = TabularDiscriminator(in_dim=in_dim, extreme_dim=extreme_dim, hidden_dim=hidden_dim)

    # 2) Adversarial trainer
    adv_trainer_1to1 = ExtremeAdversarialTrainerOneToOne(
        generator=generator_one2one,
        discriminator=discriminator_one2one,
        real_loader=train_loader,
        in_dim=in_dim,
        latent_dim=latent_dim,
        extreme_dim=extreme_dim,
        g_lr=1e-4,
        d_lr=1e-4,
        device=device
    )
    print("\n=== Start one-to-one adversarial training ===")
    adv_trainer_1to1.train(epochs=5)

    # 3) Generate adversarial samples for EACH real sample and save them
    print("\n=== Generating one-to-one adversarial samples for the entire dataset ===")
    real_data_all, fake_data_all, code_all = adv_trainer_1to1.generate_one_to_one_samples()
    print("Generated adversarial samples shape: ", fake_data_all.shape)

    # Create folder "ADsamples" if not exists
    os.makedirs("ADsamples", exist_ok=True)
    save_path = os.path.join("ADsamples", "ad_samples.csv")

    # For demonstration, save all samples in one CSV:
    # We will store columns: [real_1, real_2, ..., real_in_dim, fake_1, ..., fake_in_dim, extreme_code]
    combined_array = np.concatenate([real_data_all, fake_data_all, code_all], axis=1)
    col_names = [f"real_{i}" for i in range(in_dim)] + [f"fake_{i}" for i in range(in_dim)] + [f"extreme_code_{i}" for i in range(extreme_dim)]
    df_ad = pd.DataFrame(combined_array, columns=col_names)
    df_ad.to_csv(save_path, index=False)
    print(f"One-to-one adversarial samples saved to: {save_path}")

    # 4) Train your multi-task model as usual (optional)
    print("\n=== Training original MultiTaskModel ===")
    multi_task_model = MultiTaskModel(X_train.shape[1], Y_train.shape[1])
    trainer = EGTrainer(multi_task_model, train_loader, test_loader, train_dataset, args)
    trainer.train()

    # 5) Evaluate
    print("\n=== Load best checkpoint and evaluate ===")
    trainer.load_best_checkpoint()
    predictions, Y_hat, Y_true, events_ = trainer.predict(train_loader)
    cindex_calculator_optimized = EditedCindexOptimized()
    c11_train = cindex_calculator_optimized(Y_true, Y_hat, events_)
    print(f"C-index for Training Data: {c11_train:.4f}")

    predictions, Y_hat, Y_true, events_ = trainer.predict(test_loader)
    c11_test = cindex_calculator_optimized(Y_true, Y_hat, events_)
    print(f"C-index for Test Data: {c11_test:.4f}")
