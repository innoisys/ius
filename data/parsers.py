import re
import os.path
import pandas as pd

from glob import glob
from typing import List, Dict
from abc import ABC, abstractmethod


PARSER_REGISTRY = {

}


def register_parser(name: str):
    def decorator(cls):
        key = name.lower()
        if key in PARSER_REGISTRY:
            raise ValueError(f'Parser name {key} exists already')
        PARSER_REGISTRY[key] = cls
        return cls
    return decorator


class BaseParser:
    def __init__(self,
                 dataset_folder: str,               # e.g datasets/dataset_name/
                 mode: str,
                 image_ext: str = 'jpg',
                 **kwargs
                 ):

        self._dataset_folder = dataset_folder
        self._image_ext = image_ext

        assert mode in ['train', 'validation', 'test'], "Mode must be one of 'train', 'val', 'test'"
        self._mode = mode
        # self._kwargs = kwargs

        # For storing (x, y) items
        self._img_filenames = []            # filenames / item x in dataloader
        self._labels = []                   # labels    / item y in dataloader

        # Methods
        self.parse_dataset_folder()

        # Validity check
        assert len(self._img_filenames) == len(self._labels), \
            (f"Mismatch in number of images ({len(self._img_filenames)}) and labels ({len(self._labels)}) found "
             f"in folder {self._dataset_folder}")

        self._group_by_key = kwargs.get("group_by_key", None)
        self._group_by_value = kwargs.get("group_by_value", None)
        self._label_mapping = kwargs.get("label_mapping", None)
        self.group_dataset_by()

    @abstractmethod
    def parse_dataset_folder(self):
        raise NotImplementedError("Method must be implemented in child class")

    @property
    def image_filenames(self) -> List[str]:
        return self._img_filenames

    @property
    def labels(self) -> List[str]:
        return self._labels

    def update_image_filenames(self, filenames: List[str]) -> None:
        self._img_filenames = filenames

    def update_labels(self, labels: List[str]) -> None:
        self._labels = labels

    def group_dataset_by(self):
        if self._group_by_key is None and self._group_by_value is None:
            pass
        elif self._group_by_key is not None and self._group_by_value is not None:
            raise ValueError(
                "Please specify either 'group_key' or 'group_value' to perform group_dataset_by, not both"
            )
        else:
            if self._label_mapping is None:
                raise ValueError(
                    "No 'label_mapping' specified for perform group_dataset_by() "
                )

            filtered_filenames, filtered_labels = [], []

            if self._group_by_key is not None:
                if not isinstance(self._group_by_key, list):
                    self._group_by_key = [self._group_by_key]
                # Valid keys
                _label_mapping_keys = self._label_mapping.keys()

                # Filter (filenames & labels) if a label is in self._group_by_key and exists in label_mapping
                for _group_key in self._group_by_key:
                    if _group_key not in _label_mapping_keys:
                        raise ValueError(
                            f"{_label_mapping_keys} does not match any available label {_group_key}"
                        )
                # _valid_group_values = [self._label_mapping[_group_key] for _group_key in self._group_by_key]
                for i in range(len(self.labels)):
                    if self.labels[i] in self._group_by_key:
                        filtered_filenames.append(self._img_filenames[i])
                        filtered_labels.append(self.labels[i])

            elif self._group_by_value is not None:
                _label_mapping_values = self._label_mapping.values()
                assert self._group_by_value in _label_mapping_values, \
                    f"Label mapping dict has values {_label_mapping_values}"
                for i in range(len(self.labels)):
                    if int(self.labels[i]) == int(self._group_by_value):
                        filtered_filenames.append(self._img_filenames[i])
                        filtered_labels.append(self.labels[i])

            self.update_labels(filtered_labels)
            self.update_image_filenames(filtered_filenames)


@register_parser('filename')
class FilenameParser(BaseParser):
    """
        expected data structure:
                                datasets/dataset_name/mode
                                |
                                |----label_one*.image_ext
                                |----label_two*.image_ext
    """
    def __init__(self,
                 dataset_folder: str,               # e.g datasets/dataset_name
                 mode: str = 'train',
                 image_ext: str = 'jpg',
                 **kwargs,
                 ):

        self._label_mapping = kwargs.get("label_mapping", None)
        if self._label_mapping is None:
            raise ValueError(
                f"No 'label_mapping' specified for data parsing in FilenameParser. "
            )

        self._group_by_key = kwargs.get("group_by_key", None)
        self._group_by_value = kwargs.get("group_by_value", None)
        if self._group_by_key is None and self._group_by_value is not None:
            raise ValueError(
                "Use argument 'group_key' to perform group_dataset_by() in FilenameParser "
            )

        super().__init__(dataset_folder=dataset_folder,               # e.g dataset/train/
                         mode=mode,
                         image_ext=image_ext,
                         **kwargs)

    @staticmethod
    def extract_labels_from_file(filepath_list: List[str], label_mapping: Dict, extension: str) -> List[str]:
        unique_labels = sorted(label_mapping.keys())

        labels = []
        for filepath in filepath_list:
            filename = os.path.basename(filepath).lower()
            is_match = False
            for label_name in unique_labels:
                # {label}*.{ext} or {label}.{ext}
                pattern = rf"^{re.escape(label_name)}.*\.{re.escape(extension)}$"
                if re.match(pattern, filename):
                    label = label_name              # label = label_mapping[label_name] moved to Label Transform
                    labels.append(label)
                    is_match = True
                    break
            if not is_match:
                raise ValueError(
                    f"Filename {filepath} does not match any available label {label_mapping}"
                )
        return labels

    def parse_dataset_folder(self) -> None:
        self._img_filenames = sorted(
            glob(os.path.join(self._dataset_folder, self._mode, '*.{}'.format(self._image_ext)))
        )
        self._labels = self.extract_labels_from_file(filepath_list=self._img_filenames,
                                                     label_mapping=self._label_mapping,
                                                     extension=self._image_ext)


@register_parser("folder")
class FolderParser(BaseParser):
    """
        expected data structure:
                                datasets/dataset_name/mode
                                |----label_one
                                     |----filename_one.image_ext
                                     |----filename_two.image_ext
                                |----label_two
                                     |----filename_three.image_ext
                                     |----filename_four.image_ext
    """
    def __init__(self,
                 dataset_folder: str,               # e.g datasets/dataset_name/
                 mode: str = 'train',
                 image_ext: str = 'jpg',
                 **kwargs,
                 ):

        self._label_mapping = kwargs.get("label_mapping", None)
        if self._label_mapping is None:
            raise ValueError(
                f"No 'label_mapping' specified for data parsing in FolderParser. "
            )

        self._group_by_key = kwargs.get("group_by_key", None)
        self._group_by_value = kwargs.get("group_by_value", None)
        if self._group_by_key is None and self._group_by_value is not None:
            raise ValueError(
                "Use argument 'group_by_key' to perform group_dataset_by() in FolderParser "
            )

        super().__init__(dataset_folder=dataset_folder,               # e.g dataset/dataset_name/
                         mode=mode,
                         image_ext=image_ext,
                         **kwargs)

    def parse_dataset_folder(self) -> None:
        unique_labels = sorted(self._label_mapping.keys())
        unique_folders = sorted(os.listdir(os.path.join(self._dataset_folder, self._mode)))
        # unique_folders = [item.lower() for item in unique_folders]
        if set(unique_labels) != set(unique_folders):
            raise ValueError(
                f"Mismatch between label_mapping and folders in {os.path.join(self._dataset_folder, self._mode)}\n"
                f"Folders found: {unique_folders}\n"
                f"Labels found: {unique_labels}"
            )
        for category_folder in unique_folders:
            files = glob(os.path.join(self._dataset_folder, self._mode, category_folder, '*.{}'.format(self._image_ext)))
            self._img_filenames.extend(files)
            labels = [category_folder] * len(files)
            self._labels.extend(labels)


@register_parser("medmnist")
class MedMNISTParser(BaseParser):
    """
        expected data structure:
                                datasets/dataset_name/mode
                                |
                                |----filename_one.image_ext
                                |----filename_two.image_ext
                                |----filename_three.image_ext
                                |----filename_four.image_ext
    """
    def __init__(self,
                 dataset_folder: str,               # e.g datasets/dataset_name/
                 mode: str = 'train',
                 image_ext: str = 'jpg',
                 **kwargs,
                 ):

        self._csv_file = kwargs.pop("csv_file", None)
        if self._csv_file is None:
            raise ValueError(
                f"No 'csv_file' specified for data parsing in CSVFileParser. "
            )

        self._group_by_key = kwargs.get("group_by_key", None)
        self._group_by_value = kwargs.get("group_by_value", None)
        if self._group_by_key is not None and self._group_by_value is None:
            raise ValueError(
                "Use argument 'group_by_value' to perform group_dataset_by() in MedMNISTParser "
            )

        super().__init__(dataset_folder=dataset_folder,               # e.g dataset/train/
                         mode=mode,
                         image_ext=image_ext,
                         **kwargs)

    @staticmethod
    def read_from_medmnist_csv_file(csv_file: str, mode: str, retrieve_info: str) -> List[str]:
        df = pd.read_csv(csv_file)
        df.columns = ['split', 'filename', 'label']
        df = df[df['split'] == mode.upper()]
        return df[retrieve_info].tolist()

    def parse_dataset_folder(self) -> None:
        filenames = self.read_from_medmnist_csv_file(csv_file=self._csv_file,
                                                     mode=self._mode,
                                                     retrieve_info='filename')
        filenames = [os.path.join(self._dataset_folder, file) for file in filenames]
        self._img_filenames = filenames

        self._labels = self.read_from_medmnist_csv_file(csv_file=self._csv_file,
                                                        mode=self._mode,
                                                        retrieve_info='label')

def set_parser_class(name):
    name = name.lower()
    if name not in PARSER_REGISTRY:
        raise ValueError(
            "Unrecognized parser class name."
            f"Please use one of the following: {PARSER_REGISTRY.keys()}"
        )
    return PARSER_REGISTRY[name]