import torch
import torch.nn as nn
import numpy as np

from torch import load
from torch.utils.data import DataLoader
from typing import Union

from model.epu import EPUCNN
from utils.omega_parser import EPUCNNParams
from utils.config_utils import model_cfg_to_epucnn


class EPUCNNEval(EPUCNN):
    def __init__(self, epu_cfg: EPUCNNParams):
        super().__init__(**model_cfg_to_epucnn(epu_cfg))
        self.eval()

    # @staticmethod
    # def initialize_model_from_config(epu_cfg: EPUCNNParams, device: torch.device) -> EPUCNN:
    #     model = EPUCNN(**model_cfg_to_epucnn(epu_cfg))
    #     model.to(device)
    #     return model

    # @staticmethod
    # def load_ckpt(model: Union[nn.Module, EPUCNN], device: torch.device, ckpt_path: str) -> Union[nn.Module, EPUCNN]:
    #     state_dict = load(ckpt_path, map_location=device)
    #     model.load_state_dict(state_dict)
    #     model.to(device)
    #     model.eval()
    #     return model

    def load_ckpt(self, device: torch.device, ckpt_path: str):
        state_dict = load(ckpt_path, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)
        self.eval()
        return self

    # @staticmethod
    # def get_pretrained_model_from_config(epu_cfg: EPUCNNParams,
    #                                      device: torch.device,
    #                                      ckpt_path: str
    #                                      ):
    #     model = EPUCNNEval.initialize_model_from_config(epu_cfg, device)
    #     model = EPUCNNEval.load_ckpt(model, device, ckpt_path)
    #     return model


class InferenceRunnerEPUCNN:
    def __init__(self, epu_model: Union[EPUCNN, nn.Module], device: torch.device, mode: str = 'binary'):
        self.epu_model = epu_model
        self.device = device
        self.mode = mode

    def predict(self, dataloader: DataLoader, raw_logits=False, return_predictions: bool = False):
        self.epu_model.eval()

        all_targets = []
        all_predictions = []

        results = {}

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device, dtype=torch.float32).unsqueeze(1)  # from [bs] to [bs, 1]
                y_hat = self.epu_model(x, ret_raw_logits=raw_logits)

                all_predictions.append(y_hat.cpu().detach().numpy())
                if return_predictions:
                    all_targets.append(y.cpu().detach().numpy())

            results["predictions"] = np.concatenate(all_predictions, axis=0)
            if return_predictions:
                results["targets"] = np.concatenate(all_targets, axis=0)
            return results
