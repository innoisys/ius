import torch
import torch.nn as nn


class EarlyStopping(object):
    def __init__(self, patience: int = 10, delta: float = 0, checkpoint_path: str = 'checkpoint.pt',
                 verbose: bool = True, restore_best_weights: bool = True, monitor: str = 'val_loss',
                 mode: str = 'min'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement before stopping.
            delta (float):  Minimum change to qualify as an improvement.
            checkpoint_path (str): File path to save the best model.
            verbose (bool): If True, prints messages when improvement occurs.
            restore_best_weights (bool): If True, loads the best weights saved during training.
            monitor (str): Metric name to monitor (e.g., 'val_loss', 'val_f1', 'val_accuracy').
            mode (str): 'min' if lower is better (e.g. loss), 'max' if higher is better (e.g. accuracy, f1).
        """

        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.restore_best_weights_flag = restore_best_weights
        self.monitor = monitor
        self.mode = mode

        if self.mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")

        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.best_value = float('inf') if mode == 'min' else -float('inf')

    def __call__(self, current_value: float, model: nn.Module, epoch: int = None):
        improvement = (
                (self.mode == 'min' and current_value < self.best_value - self.delta) or
                (self.mode == 'max' and current_value > self.best_value + self.delta)
        )

        if self.best_score is None:
            self.best_score = current_value
            self.best_epoch = epoch if epoch is not None else 0
            self.save_checkpoint(current_value, model)
            return False

        if improvement:
            if self.verbose:
                print(f"EarlyStopping (Improved): {self.monitor} {self.best_value:.4f} → {current_value:.4f}")
            self.best_score = current_value
            self.best_epoch = epoch if epoch is not None else 0
            self.save_checkpoint(current_value, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping (No Improvement): {self.counter}/{self.patience} epochs "
                      f"({self.monitor}={current_value:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, current_value: float, model: nn.Module):
        if self.verbose:
            print(f"Saving new best model at {self.checkpoint_path} ({self.monitor}={current_value:.4f})")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.best_value = current_value

    def restore_best_weights(self, model: nn.Module):
        if self.restore_best_weights_flag:
            if torch.cuda.is_available():
                map_location = torch.device('cuda')
            else:
                map_location = torch.device('cpu')

            try:
                model.load_state_dict(torch.load(self.checkpoint_path, map_location=map_location))
                if self.verbose:
                    print(f'Restored best model from {self.checkpoint_path} (epoch {self.best_epoch})')
            except FileNotFoundError:
                if self.verbose:
                    print(f'Restoring failed | No checkpoint found at {self.checkpoint_path}, cannot restore weights.')


class EarlyStoppingCallback(object):
    """Adapter to use EarlyStopping as a callback in the trainer function."""

    def __init__(self, patience=10, delta=0, checkpoint_path='checkpoint.pt', verbose=True, restore_best_weights=True,
                 monitor='val_loss', mode='min', save_final_model=False,
                 ):
        self.early_stopping = EarlyStopping(
            patience=patience,
            delta=delta,
            checkpoint_path=checkpoint_path,
            verbose=verbose,
            restore_best_weights=restore_best_weights,
            monitor=monitor,
            mode=mode
        )
        self.checkpoint_path = checkpoint_path
        self.save_final_model = save_final_model

    def on_training_begin(self, state):
        state['early_stop'] = False
        state['best_epoch'] = 0
        state['best_model_path'] = self.checkpoint_path

    def on_validation_end(self, state):
        epoch = state.get('epoch', None)
        metric_name = self.early_stopping.monitor
        # print(state.keys())
        # print(state.get('val_metrics'))

        if metric_name.startswith('val_metrics'):
            key = metric_name.split('.')[-1]
            metric_value = state['val_metrics'].get(key)
        elif metric_name.startswith('train_metrics'):
            key = metric_name.split('.')[-1]
            metric_value = state['train_metrics'].get(key)
        else:
            metric_value = state.get(metric_name)

        if metric_value is None:
            raise ValueError(f"Metric '{metric_name}' not found in state dictionary.")

        stop = self.early_stopping(metric_value, state['model'], epoch=epoch)

        state['early_stop'] = stop
        state['best_epoch'] = self.early_stopping.best_epoch
        state['best_val_loss'] = self.early_stopping.best_value

    def on_training_end(self, state):
        model = state.get('model')
        if model:
            self.early_stopping.restore_best_weights(model)
            print(f"Training finished. Best epoch: {self.early_stopping.best_epoch}, "
                  f"Best {self.early_stopping.monitor}: {self.early_stopping.best_value:.4f}")

            if self.save_final_model:
                final_path = self.checkpoint_path.replace(".pt", "_final.pt")
                torch.save(model.state_dict(), final_path)
                print(f"Final model saved at {final_path}")