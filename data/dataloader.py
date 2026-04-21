from typing import Union
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.omega_parser import DataLoading
from data.dataset import EPUDataset
from data.loading import EPUDatasetFromConfig


def to_dataloader(dataset: Union[Dataset, EPUDataset, EPUDatasetFromConfig],
                  loading_cfg: DataLoading) -> DataLoader:

    return DataLoader(dataset,
                      batch_size=loading_cfg.batch_size,
                      shuffle=loading_cfg.shuffle,
                      num_workers=loading_cfg.num_workers,
                      pin_memory=loading_cfg.pin_memory,
                      persistent_workers=loading_cfg.persistent_workers,
                      drop_last=False
                      )
