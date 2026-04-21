import numpy as np
import torch
import torch.nn as nn

from typing import Optional, Dict, Tuple, Union, Callable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class EPUMetrics:

    def __init__(self,
                 mode:              str = "binary",
                 n_classes:         int = 1,
                 confidence_level:  float = 0.5,
                 activation:        Union[nn.Module, Callable] = nn.Sigmoid(),
                 metrics_config:    Optional[Dict[str, bool]] = None,
                 ):

        assert mode in ["binary", "multiclass"], "mode should be either 'binary' or 'multiclass'"
        self.mode = mode
        
        self.n_classes = n_classes
        self.confidence = confidence_level

        self.activation = activation

        default_metrics = {"accuracy": True,  "auc": True}
        self.metrics_config = metrics_config if metrics_config is not None else default_metrics

        self.avg_method = 'binary' if self.mode == "binary" else 'macro'

    @staticmethod
    def _to_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x = x.detach()
            return x
        else:
            return torch.as_tensor(x)

    @staticmethod
    def _to_numpy(x: torch.Tensor) -> np.array:
        x = x.detach().cpu().numpy()
        return x

    def _apply_activation(self, raw_logits: torch.Tensor) -> torch.Tensor:
        return self.activation(raw_logits)

    def _format_labels(self,
                       y_true: torch.Tensor,
                       y_pred: torch.Tensor,
                       y_prob: torch.Tensor = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if y_prob is None:
            y_prob = self._apply_activation(y_pred)

        if self.mode == 'binary':
            y_true = y_true.view(-1)
            y_prob = y_prob.view(-1)
            y_pred = (y_prob > self.confidence).to(dtype=torch.int64)

        else:
            if y_true.ndim > 1:
                y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)

        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        y_prob = self._to_numpy(y_prob)
        return y_true, y_pred, y_prob

    @staticmethod
    def convert_to_onehot(labels: np.ndarray, n_classes: int) -> np.ndarray:
        if n_classes is None:
            raise ValueError("Number of classes is not provided")
        labels = labels.astype(int)
        labels = np.eye(n_classes)[labels]
        return labels

    def compute(self,
                y_true: Union[torch.Tensor, np.ndarray],
                y_pred: Union[torch.Tensor, np.ndarray],
                y_prob: Optional[Union[torch.Tensor, np.ndarray]] = None
                ) -> Dict[str, float]:
        y_true = self._to_tensor(y_true)
        y_pred = self._to_tensor(y_pred)
        y_prob = None if y_prob is None else self._to_tensor(y_prob)

        y_true_np, y_pred_np, y_prob_np = self._format_labels(
            y_true=y_true,
            y_pred=y_pred,          # raw_logits
            y_prob=y_prob)

        metrics = {}
        if self.metrics_config["accuracy"]:
            score = accuracy_score(y_true_np, y_pred_np)
            metrics["accuracy"] = float(np.round(score, 4))
        if self.metrics_config["auc"]:
            try:
                if self.avg_method == "binary":
                    score = roc_auc_score(y_true_np, y_prob_np)
                else:
                    y_true_np = self.convert_to_onehot(y_true_np, n_classes=self.n_classes)
                    score = roc_auc_score(y_true_np, y_prob_np, multi_class='ovr')
                metrics["auc"] = score
            except ValueError:
                metrics["auc"] = float('nan')       # AUC cannot be calculated

        return metrics

    def update_confidence(self, confidence_level: float) -> None:
        self.confidence = confidence_level
