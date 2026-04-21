import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from typing import List, Any, Optional, Tuple, Dict

# mine
from data.dataloader import DataLoader


class EPUTrainer:
    def __init__(self,
                 model:         nn.Module,
                 device:        torch.device,
                 optimizer:     optim.Optimizer,
                 criterion:     nn.Module,
                 epochs:        int,
                 train_loader:  DataLoader,
                 val_loader:    Optional[DataLoader] = None,
                 callbacks:     Optional[List[object]] = None,
                 metrics:       Optional = None,
                 checkpoint_dir:    Optional[str] = None,
                 ):
        self.model = model
        self.val_loader = val_loader
        self.train_loader = train_loader

        self.device = device
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks or []
        self.checkpoint_dir = checkpoint_dir

        self.metrics_fun = metrics
        # if self.metrics_fun is None:

        # init values
        self.best_metric = float("inf")
        self.best_model_path = None
        self.history = []

        self.state = {"model": self.model,
                      "epoch": 0,
                      "early_stop": False,
                      }

    def train(self):
        self.model.to(self.device)

        self._on_training_begin()

        for epoch in range(self.epochs):
            self.state["epoch"] = epoch
            self._on_epoch_begin()

            train_loss, train_metrics = self._train_one_epoch()
            val_loss, val_metrics = self._validate_epoch()

            self.history.append({"epoch": epoch,
                                 "train_loss": train_loss,
                                 "val_loss": val_loss,
                                 "train_metrics": train_metrics,
                                 "val_metrics": val_metrics,}
                                )

            self._on_epoch_end(train_loss, train_metrics, val_loss, val_metrics)
            self._on_validation_end()

            if self.state.get("early_stop", False):
                print("Early stopping triggered.")
                break

        self._on_training_end()
        # self._export_metrics_to_json()

    def _train_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        running_loss = 0.0
        predictions, ground_truth = [], []

        for i, sample in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {self.state['epoch'] + 1}")):
            x, y = sample
            x = x.to(self.device)
            y = y.to(self.device, dtype=torch.float32).unsqueeze(1)     # from [bs] to [bs, 1]

            self.optimizer.zero_grad()

            y_hat = self.model(x, ret_raw_logits=True)                 # w/o EPU activation -applied internally in loss
            loss = self.criterion(y_hat, y)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            predictions.append(y_hat.detach().cpu())
            ground_truth.append(y.detach().cpu())

            for callback in self.callbacks:
                if hasattr(callback, "on_batch_end"):
                    callback.on_batch_end(
                        {**self.state,
                         "batch": i,
                         "loss": loss.item()}
                    )

        avg_loss = running_loss / len(self.train_loader)

        metrics = {}
        if self.metrics_fun is not None:
            metrics = self.metrics_fun.compute(
                y_true=torch.cat(ground_truth, axis=0),
                y_pred=torch.cat(predictions, axis=0)
            )
        return avg_loss, metrics

    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        if self.val_loader is None:
            return 0.0, {}

        self.model.eval()
        total_loss = 0
        predictions, ground_truths = [], []

        with torch.no_grad():
            for sample in tqdm(self.val_loader, desc="Validating"):
                x, y = sample
                x = x.to(self.device)
                y = y.to(self.device, dtype=torch.float32).unsqueeze(1)  # from [bs] to [bs, 1]
                y_hat = self.model(x, ret_raw_logits=True)
                loss = self.criterion(y_hat, y)

                total_loss += loss.item()
                predictions.append(y_hat.detach().cpu())
                ground_truths.append(y.detach().cpu())

        avg_loss = total_loss / len(self.val_loader)
        metrics = {}
        if self.metrics_fun is not None:
            metrics = self.metrics_fun.compute(
                y_true=torch.cat(ground_truths, axis=0),
                y_pred=torch.cat(predictions, axis=0)
            )

        return avg_loss, metrics

    def _on_training_begin(self):
        for callback in self.callbacks:
            if hasattr(callback, "on_training_begin"):
                callback.on_training_begin(self.state)

    def _on_epoch_begin(self):
        for callback in self.callbacks:
            if hasattr(callback, "on_epoch_begin"):
                callback.on_epoch_begin(self.state)

    def _on_epoch_end(self, train_loss, train_metrics, val_loss, val_metrics):
        # update state
        self.state.update(
            {"train_loss": train_loss,
             "val_loss": val_loss,
             "train_metrics": train_metrics,
             "val_metrics": val_metrics,
             }
        )
        # print losses
        print(f"Epoch {self.state['epoch'] + 1} | "
              f"Train loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        # print metrics
        if train_metrics is not None:
            train_metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            print(f"Train metrics:\t\t {train_metrics_str}")
        if val_metrics:
            val_metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            print(f"Validation metrics:\t {val_metrics_str}")

        # exec callbacks
        for callback in self.callbacks:
            if hasattr(callback, "on_epoch_end"):
                callback.on_epoch_end(self.state)

    def _on_validation_end(self,):
        for callback in self.callbacks:
            if hasattr(callback, "on_validation_end"):
                # print(self.state)
                callback.on_validation_end(self.state)

    def _on_training_end(self):
        for callback in self.callbacks:
            if hasattr(callback, "on_training_end"):
                callback.on_training_end(self.state)

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_metrics(self):
        return self.metrics_fun

    # def _export_metrics_to_json(self):
    #     if self.checkpoint_dir is not None:
    #         metrics_path = os.path.join(self.checkpoint_dir, "metrics.json")
    #         with open(metrics_path, "w") as f:
    #             json.dump(self.history, f, indent=4)
    #         print(f"Metrics exported to {metrics_path}")
