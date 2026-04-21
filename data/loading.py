# from torch.utils.data import Dataset
from typing import Union

from data.data_utils import LabelTransform
from utils.omega_parser import DataParams
from data.parsers import set_parser_class
from data.perceptual_transforms import PerceptualFeatureMapTransform
from data.dataset import EPUDataset
from utils.config_utils import data_cfg_to_dataparser
from ius.ius_eval_parser import IUSEvalParser


# Creates an EPUDataset from IUS config.
# Used in EPUCNN training, classification performance eval & calculation of cb vectors
class EPUDatasetFromConfig:
    def __init__(self, dataconfig: DataParams, **kwargs):

        self.dataset_path = dataconfig.dataset_path
        self.images_extension = dataconfig.images_extension

        self.data_preprocessing = dataconfig.data_preprocessing
        # self.data_loading = dataconfig.data_loading

        self.group_by = kwargs.get('group_by')

    def get_dataset(self, dataset_mode: str) -> EPUDataset:
        assert dataset_mode in ["train", "validation", "test"], "Dataset mode must be either train or val or test."

        # Create parser & transforms for Dataset
        parser = set_parser_class(
            name=self.data_preprocessing.data_parser)(
            **data_cfg_to_dataparser(
                dataset_path=self.dataset_path,
                images_extension=self.images_extension,
                data_mode=dataset_mode,
                preprocessing_cfg=self.data_preprocessing,
                group_by=self.group_by,
            )
        )

        perceptual_transform = PerceptualFeatureMapTransform(
            resize_dims=self.data_preprocessing.resize_dims,
            resize_mode="bicubic",
            data_mode=self.data_preprocessing.data_mode
        )

        label_transform = LabelTransform(
            mapping_dict=self.data_preprocessing.label_mapping
        )

        # Create Dataset
        dataset = EPUDataset(
            data_parser=parser,
            perceptual_transform=perceptual_transform,
            label_transform=label_transform
        )
        return dataset


# used during IUS evaluation
class IUSEvalDataset:
    def __init__(self, dataconfig: DataParams, **kwargs):

        self.dataset_path = dataconfig.dataset_path
        self.images_extension = dataconfig.images_extension

        self.data_preprocessing = dataconfig.data_preprocessing
        # self.data_loading = dataconfig.data_loading

        self.group_by = kwargs.get('group_by')

    def get_dataset(self, parser: Union[IUSEvalParser]) -> EPUDataset:

        perceptual_transform = PerceptualFeatureMapTransform(
            resize_dims=self.data_preprocessing.resize_dims,
            resize_mode="bicubic",
            data_mode=self.data_preprocessing.data_mode
        )

        # Create Dataset
        dataset = EPUDataset(
            data_parser=parser,
            perceptual_transform=perceptual_transform,
            label_transform=None
        )
        return dataset
