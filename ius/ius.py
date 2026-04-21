import os
import re
import torch
import numpy as np
import torch.nn.functional as F

from glob import glob
from typing import Optional, Union, List
from pathlib import Path


class IUS:
    def __init__(self,
                 cb_path: str,
                 cb_tag: Optional[str] = None,
                 device: Optional[Union[str, torch.device]] = None,):

        assert cb_path.endswith('.npy') or Path(cb_path).is_dir(), \
            'cb_path should be an .npy file or a directory with .npy files '

        # cb_path is an .npy file
        if cb_path.endswith('.npy') and not Path(cb_path).is_dir():
            self.cb_vector = np.load(cb_path)

            if cb_tag is None:
                # extract from .npy filename
                match = re.match(r"cb_vector_(.*)\.npy", os.path.basename(cb_path))
                assert match is not None, f"Could not extract cb_tag from filename: {cb_path}"
                cb_tag = match.group(1)

            self.cb_tag = cb_tag
            self.cb_mapping = {str(self.cb_tag): self.cb_vector}
            self.cb_has_value = True
            print("cb_vector loaded !")
        else:
            # cb_path is a dir with .npy files
            self._load_cb_vecs_from_path(cb_path)

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_cb_vecs_from_path(self, cb_path: str):
        cb_files = glob(os.path.join(cb_path, 'cb_vector_*.npy'))
        self.cb_mapping = {}

        for cb_file in cb_files:
            cb_vector = np.load(cb_file)
            match = re.match(r"cb_vector_(.*)\.npy", os.path.basename(cb_file))
            if match is None:
                continue
            tag = match.group(1)
            self.cb_mapping[tag] = cb_vector

        self.cb_vector = None
        self.cb_tag = None
        self.cb_has_value = False

    def update_cb_vector(self, cb_vector: Union[torch.Tensor, np.ndarray]):
        self.cb_vector = cb_vector

    def update_cb_tag(self, cb_tag: str):
        self.cb_tag = cb_tag

    def update_cb_mapping(self, cb_vector: Union[torch.Tensor, np.ndarray], cb_tag: str):
        self.cb_mapping[cb_tag] = cb_vector

    def update_device(self, device: Union[str, torch.device]):
        self.device = device

    def update(self, cb_vector: Union[torch.Tensor, np.ndarray], cb_tag: str):
        self.update_cb_vector(cb_vector)
        self.update_cb_tag(cb_tag)
        self.cb_has_value = True
        
        # add cb_vector to cb_mapping if is a new one
        existing_keys = list(self.cb_mapping.keys())
        if cb_tag not in existing_keys:
            self.update_cb_mapping(cb_vector, cb_tag)

    @staticmethod
    def _ius_score(cb_vector: torch.Tensor, c_vector: torch.Tensor) -> float:
        ius_score = F.cosine_similarity(cb_vector, c_vector, dim=0)  # cb_vec/c_vec of shape [num_subnets, fc_pred_units]
        ius_score = ius_score.item()
        return float(ius_score)

    @staticmethod
    def calculate_ius_across_multiple_c(cb_vector: Union[torch.Tensor, np.ndarray],
                                        c_vectors: torch.Tensor,
                                        device: Union[str, torch.device]) -> List[float]:
        ius_scores = []

        if not isinstance(cb_vector, torch.Tensor):
            cb_vector = torch.as_tensor(cb_vector)      # [num_subnets, fc_pred_units]
        # cb_vector = cb_vector.to(c_vectors.device)
        cb_vector = cb_vector.to(device, dtype=c_vectors.dtype)

        num_of_c_vecs = c_vectors.shape[0]              # [fake_images, num_subnets, fc_pred_units]
        for i in range(num_of_c_vecs):
            c_hat_vec = c_vectors[i]
            c_hat_vec = c_hat_vec.to(device)
            ius_score = IUS._ius_score(cb_vector, c_hat_vec)
            ius_scores.append(ius_score)
        return ius_scores

    def ius_measure(self, c_vectors: torch.Tensor, cb_tag: Optional[str] = None) -> List[float]:
        #  IUS init from cb_path that is a dir with multiple .npy & cb_tag is still None
        if not self.cb_has_value:
            assert cb_tag is not None, 'Please provide cb_tag'

            existing_tags = self.cb_mapping.keys()
            assert cb_tag in existing_tags, f'cb_tag provided is not available. Available cb_tags: {existing_tags}'

            self.update_cb_tag(cb_tag)
            cb_vector = self.cb_mapping[self.cb_tag]
            self.update_cb_vector(cb_vector)
            self.cb_has_value = True

        # IUS init from .npy cb_path
        print(f"Calculating IUS score for cb_tag = {self.cb_tag}")
        ius_scores = self.calculate_ius_across_multiple_c(cb_vector=self.cb_vector,
                                                          c_vectors=c_vectors,
                                                          device=self.device)
        return ius_scores



