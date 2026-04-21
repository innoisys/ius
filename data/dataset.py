from torch.utils.data import Dataset

from .data_utils import load_image

from .parsers import BaseParser
from .data_utils import LabelTransform
from .perceptual_transforms import PerceptualFeatureMapTransform


class EPUDataset(Dataset):
    def __init__(self,
                 data_parser: BaseParser,
                 perceptual_transform: PerceptualFeatureMapTransform = None,
                 label_transform: LabelTransform = None,
                 **kwargs):

        self.data_parser = data_parser
        self.transform = perceptual_transform
        self.label_transform = label_transform
        self.kwargs = kwargs

        self.image_paths = self.data_parser.image_filenames
        self.labels = self.data_parser.labels

    def __len__(self):
        assert len(self.image_paths) == len(self.labels), "Mismatch in image paths and labels"
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = load_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return img, label
