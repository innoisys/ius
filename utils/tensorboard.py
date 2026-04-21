import os
import time
import subprocess
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger(object):
    def __init__(self, log_dir='runs/experiment'):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    # def log_scalars(self, tag: str, values: dict, step: int):
    #     self.writer.add_scalars(tag, values, step)

    def log_histogram(self, model:nn.Module, step: int):
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param, step)
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, step)

    def log_model_graph(self, model: nn.Module, input_sample: torch.Tensor):
        self.writer.add_graph(model, input_sample)

    def close(self):
        self.writer.close()


def launch_tensorboard(log_dir: str = "runs", port: int = 6006, open_browser: bool = True):
    """Automatically launch TensorBoard pointing to log_dir."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    print(f"[TensorBoard] launching at http://localhost:{port}/")

    tb_command = ["tensorboard", f"--logdir={log_dir}", f"--port={port}"]
    if not open_browser:
        tb_command.append("--host=127.0.0.1")

    subprocess.Popen(tb_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)


class TensorboardLoggerCallback(object):
    """
            Adapter uses TensorboardLogger
    """
    def __init__(self,
                 log_dir: str = 'logs/test_experiment',
                 log_histograms: bool = False,
                 launch_tb: bool = True,
                 tb_port: int = 6006,
                 open_tb_in_browser: bool = False,
                 ):

        self.tb_logger = TensorboardLogger(log_dir=log_dir)

        self.log_histograms = log_histograms

        if launch_tb:
            launch_tensorboard(log_dir=log_dir,
                               port=tb_port,
                               open_browser=open_tb_in_browser)

        self.model = None

    def on_train_begin(self, model: nn.Module):
        self.model = model
        print("[TensorBoard] Training started, callback init")

    def on_train_end(self):
        self.tb_logger.close()
        print("[TensorBoard] Training ended, logs saved")

    def on_epoch_end(self, state):
        epoch = state.get('epoch', 0)

        # losses
        self.tb_logger.log_scalar('Loss/train', state.get('train_loss', 0.0), epoch)
        self.tb_logger.log_scalar('Loss/val', state.get('val_loss', 0.0), epoch)

        # training & validation metrics

        # for metric, value in train_metrics.items():
        #     self.tb_logger.log_scalar(f'Metrics/train/{metric}', value, epoch)
        train_metrics = state.get('train_metrics')
        if train_metrics is not None:
            # self.tb_logger.log_scalars(f'Metrics/train', train_metrics, epoch)
            for metric_name, metric_value in train_metrics.items():
                self.tb_logger.log_scalar(f'Metrics/train/{metric_name}', metric_value, epoch)
        val_metrics = state.get('val_metrics')
        if val_metrics is not None:
            for metric_name, metric_value in val_metrics.items():
                self.tb_logger.log_scalar(f'Metrics/val/{metric_name}', metric_value, epoch)

        # weights and grads
        model = state.get('model')
        if self.log_histograms and model is not None:
            self.tb_logger.log_histogram(model, epoch)

