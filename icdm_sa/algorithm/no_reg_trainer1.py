import os.path
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import grad
from tqdm import trange
from torch.nn.utils import clip_grad_norm_
from .util import binarize_and_sum_columns
from pathlib import Path

class NoRegularizationTrainer:
    def __init__(self, model, train_loader, test_loader, optimizer, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, min_lr=0.00001)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = args.epochs
        self.clip = args.clip

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop_patience = args.early_stop_patience
        self.checkpoint_path = os.path.join(args.save_path, "best_model.pth")
        self.args = args
        self.gradient_importance = None
        
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        Path(self.args.save_path).mkdir(exist_ok=True, parents=True)
        torch.save(state, os.path.join(self.args.save_path, filename))
        if is_best:
            torch.save(state, os.path.join(self.args.save_path, "best_model.pth"))

    def train(self):
        for epoch in trange(self.epochs):
            self.model.train()
            running_loss = 0.0
            gradient_importance = None  # 初始化梯度累积变量

            for batch_idx, (X_train, targets, masks, event_train, _) in enumerate(self.train_loader):
                X_train = X_train.to(self.device)
                X_train.requires_grad = True
                targets = [target.to(self.device) for target in targets]
                masks = [mask.to(self.device) for mask in masks]

                # 前向传播
                task_outputs = self.model(X_train)
                loss = self.model.custom_loss(task_outputs, targets, masks)
                self.optimizer.zero_grad()

                # 反向传播
                loss.backward(retain_graph=True)

                # 累积梯度
                if gradient_importance is None:
                    gradient_importance = X_train.grad.sum(dim=0)
                else:
                    gradient_importance += X_train.grad.sum(dim=0)

                # 更新模型参数
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"End of Epoch {epoch}, Average Training Loss: {avg_loss:.4f}")

            # 验证
            early_stop = self.validate(epoch)

            # 如果触发早停或到最后一个 epoch，保存梯度重要性
            if epoch == self.epochs - 1 or early_stop:
                if gradient_importance is not None:
                    self.gradient_importance = gradient_importance / len(self.train_loader.dataset)
                    print(f"Gradient importance saved at epoch {epoch}.")
                else:
                    print("Warning: No gradient importance was calculated.")
                    self.gradient_importance = torch.zeros(X_train.size(1), device=self.device)

            # 提前退出
            if early_stop:
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
        all_masks = np.zeros((len(data_loader) * self.args.batch_size, self.model.num_tasks))
        events = np.zeros((len(data_loader) * self.args.batch_size))
        # Disable gradient calculations
        with torch.no_grad():
            for i, (X, targets, masks, status, _) in enumerate(data_loader):
                # Forward pass
                X = X.to(self.device)
                task_outputs_ = self.model(X)
                events[self.args.batch_size * i: (self.args.batch_size * (i + 1))] = status.cpu().numpy()
                for j, task_output in enumerate(task_outputs_):
                    predictions[self.args.batch_size * i: (self.args.batch_size * (i + 1)), j] = \
                        task_output.cpu().numpy()[:, 0]
                    Y_true[self.args.batch_size * i: (self.args.batch_size * (i + 1)), j] = \
                        targets[j].cpu().numpy()[:, 0]
                    all_masks[self.args.batch_size * i: (self.args.batch_size * (i + 1)), j] = \
                        masks[j].cpu().numpy()[:, 0]
        # Y_true = np.multiply(Y_true, all_masks)
        Y_true = np.sum(Y_true[:, :], axis=1)
        Y_hat = (predictions > 0.5).astype(int)
        Y_hat = np.sum(Y_hat[:, :], axis=1)
        return predictions, Y_hat, Y_true, events

    # def predict(self, data_loader):
    #     self.model.eval()
    #     predictions = np.zeros((len(data_loader) * self.args.batch_size, self.model.num_tasks))
    #     Y_true = np.zeros((len(data_loader) * self.args.batch_size, self.model.num_tasks))
    #     all_masks = np.zeros((len(data_loader) * self.args.batch_size, self.model.num_tasks))
    #     events = np.zeros((len(data_loader) * self.args.batch_size))
    #     # Disable gradient calculations
    #     with torch.no_grad():
    #         for i, (X, targets, masks, status) in enumerate(data_loader):
    #             # Forward pass
    #             X = X.to(self.device)
    #             task_outputs_ = self.model(X)
    #             events[self.args.batch_size * i: (self.args.batch_size * (i + 1))] = status.cpu().numpy()
    #             for j, task_output in enumerate(task_outputs_):
    #                 predictions[self.args.batch_size * i: (self.args.batch_size * (i + 1)), j] = \
    #                     task_output.cpu().numpy()[:, 0]
    #                 Y_true[self.args.batch_size * i: (self.args.batch_size * (i + 1)), j] = \
    #                     targets[j].cpu().numpy()[:, 0]
    #                 all_masks[self.args.batch_size * i: (self.args.batch_size * (i + 1)), j] = \
    #                     masks[j].cpu().numpy()[:, 0]
    #
    #     for i in range(1, predictions.shape[1]):
    #         predictions[:, i] = np.multiply(predictions[:, i-1] , predictions[:, i])
    #     Y_true = np.multiply(Y_true, all_masks)
    #     Y_true = np.sum(Y_true, axis=1)
    #     Y_hat = (predictions > 0.5).astype(int)
    #     Y_hat = np.sum(Y_hat, axis=1)
    #     return predictions, Y_hat, Y_true, events

