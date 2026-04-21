import os
import torch
import numpy as np
import torch.nn.functional as F

from glob import glob
from pathlib import Path
from typing import List, Union

from data.parsers import BaseParser


VALID_EXT = [".png", ".jpg", ".jpeg", ".tiff", ".tif",]


class IUSEvalParser(BaseParser):
    """
        reads files inside a folder and creates a dummy parser (files, None)
        labels are not required for IUS evaluation.
        in the case where path is a single filename it creates a dummy parser (filename, None)
    """

    def __init__(self,
                 path: str,               # e.g datasets/dataset_name
                 image_ext: str = 'jpg',
                 **kwargs,
                 ):

        self.single_item = False
        self.synthetic_images_path = path

        if Path(path).suffix.lower() in VALID_EXT and not Path(path).is_dir():
            self.single_item = True
            dataset_folder = str(Path(path).parent)
        else:
            dataset_folder = self.synthetic_images_path

        super().__init__(dataset_folder=dataset_folder,               # e.g path_to_fake_images
                         mode='test',
                         image_ext=image_ext,
                         **kwargs)

    def parse_dataset_folder(self) -> None:
        if self.single_item:
            self._img_filenames = [self.synthetic_images_path]              # [self._dataset_folder]
        else:
            self._img_filenames = sorted(glob(os.path.join(self.synthetic_images_path, f"*.{self._image_ext}")))
        self._labels = [-1 for _ in range(len(self._img_filenames))]       # ignored

