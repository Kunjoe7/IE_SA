import os.path
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import grad
from tqdm import trange
from .expected_gradient_multi_task import AttributionPriorExplainer
from .util import binarize_and_sum_columns
from torch.nn.utils import clip_grad_norm_
import numpy as np

class EGTrainer:
    def __init__(self, model, train_loader, test_loader, train_dataset, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        self.lambda_reg = args.lambda_reg
        self.APExp = AttributionPriorExplainer(train_dataset, args.batch_size, k=args.eg_k)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = args.epochs
        self.clip = args.clip

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop_patience = args.early_stop_patience
        self.checkpoint_path = os.path.join(args.save_path, "best_model.pth")
        self.args = args

        # 初始化 attribution 矩阵
        self.attribution_matrix = None

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        torch.save(state, os.path.join(self.args.save_path, filename))
        if is_best:
            torch.save(state, os.path.join(self.args.save_path, "best_model.pth"))

    def train(self):
        for epoch in trange(self.epochs):
            self.model.train()
            running_loss = 0.0
            gradient_norms = []
            attribution_cumulative = None  # 用于累积所有批次的 attributions

            for batch_idx, (X_train, targets, masks, event_train, _) in enumerate(self.train_loader):
                X_train = X_train.to(self.device)
                # X_train.requires_grad = True
                targets = [target.to(self.device) for target in targets]
                masks = [mask.to(self.device) for mask in masks]

                self.optimizer.zero_grad()
                task_outputs = self.model(X_train)

                # 获取 attribution
                eg = self.APExp.shap_values(self.model, X_train)

                # 累积 attributions
                batch_attribution = torch.stack([torch.mean(eg_task, dim=0) for eg_task in eg], dim=0)  # 每个任务的平均值
                if attribution_cumulative is None:
                    attribution_cumulative = batch_attribution
                else:
                    attribution_cumulative += batch_attribution

                # 计算损失
                L_base = self.model.custom_loss(task_outputs, targets, masks)

                Omega_smooth = 0
                prev_attributions = None
                for t in range(self.model.num_tasks):
                    current_attributions = eg[t]
                    if prev_attributions is not None:
                        diff = current_attributions - prev_attributions
                        Omega_smooth += torch.abs(diff)
                    prev_attributions = current_attributions

                Omega_smooth = torch.sum(Omega_smooth)
                loss = L_base + (self.lambda_reg * Omega_smooth)
                loss.backward(retain_graph=True)

                total_norm = 0
                for param in self.model.parameters():
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                gradient_norms.append(total_norm)

                clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            avg_grad_norm = sum(gradient_norms) / len(gradient_norms)
            print(
                f'End of Epoch {epoch}, Average Training Loss: {avg_loss:.4f}, Average Gradient Norm: {avg_grad_norm:.4f}')

            # 在最后一个 epoch 或触发早停时保存 attribution
            if epoch == self.epochs - 1 or self.validate(epoch):
                self.attribution_matrix = (attribution_cumulative / len(self.train_loader)).detach()  # 去掉梯度
                print(f"Attribution matrix saved at epoch {epoch}.")
                return

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, val_targets, val_masks, event_val, _ in self.test_loader:
                X_val = X_val.to(self.device)
                val_targets = [target.to(self.device) for target in val_targets]
                val_masks = [mask.to(self.device) for mask in val_masks]
                val_outputs = self.model(X_val)
                loss = self.model.custom_loss(val_outputs, val_targets, val_masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.test_loader)
        print(f'End of Epoch {epoch}, Average Validation Loss: {avg_val_loss:.4f}')

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
        # Restore the model and optimizer state
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # Optionally, restore other variables such as epoch and best validation loss
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded model from checkpoint at epoch {start_epoch} with best validation loss {best_val_loss:.4f}")

    def predict(self, data_loader):
        self.model.eval()
        predictions = np.zeros((len(data_loader) * self.args.batch_size, self.model.num_tasks))
        Y_true = np.zeros((len(data_loader) * self.args.batch_size, self.model.num_tasks))
        events = np.zeros((len(data_loader) * self.args.batch_size))
        # Disable gradient calculations
        with torch.no_grad():
            for i, (X, targets, _, status, _) in enumerate(data_loader):
                # Forward pass
                X = X.to(self.device)
                task_outputs_ = self.model(X)
                events[self.args.batch_size * i: (self.args.batch_size * (i + 1))] = status.cpu().numpy()
                for j, task_output in enumerate(task_outputs_):
                    predictions[self.args.batch_size * i: (self.args.batch_size * (i + 1)), j] = \
                        task_output.cpu().numpy()[:, 0]
                    Y_true[self.args.batch_size * i: (self.args.batch_size * (i + 1)), j] = \
                        targets[j].cpu().numpy()[:, 0]
        Y_true = np.sum(Y_true, axis=1)
        Y_hat = (predictions > 0.5).astype(int)
        Y_hat = np.sum(Y_hat, axis=1)
        return predictions, Y_hat, Y_true, events


