from PIL import Image
from typing import Dict


def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    return img


class LabelTransform:
    def __init__(self, mapping_dict: Dict):
        self.mapping_dict = mapping_dict
        self.mapping_dict = {k.lower(): v for k, v in mapping_dict.items()}
        self._keys = sorted(self.mapping_dict.keys())
        # self._keys = [item.lower() for item in self._keys]

    def __call__(self, label):
        label = label.lower()
        assert label in self._keys, (f'label {label} not in label mapping_dict provided.'
                                     f'Available keys {self._keys}')
        return self.mapping_dict[label]

        