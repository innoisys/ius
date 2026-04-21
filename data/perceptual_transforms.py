import torch
import numpy as np
import torch.nn.functional as F

import PIL.Image as Image
import kornia.color as Kcolor
import kornia.filters as Kfilters

from typing import Tuple


def to_chw_tensor(img: Image) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(np.array(img), dtype=torch.float32)
    # img = img.float()

    # Image.mode('L')
    if img.ndim not in (2, 3):
        raise ValueError(f'Image shape {img.shape} is not supported')
    elif img.ndim == 2:
        img = img.unsqueeze(0)              # [1, h, w]
    elif img.ndim == 3:
        img = img.permute(2, 0, 1)          # [c, h, w]

    if img.shape[0] not in (1, 3):
        raise ValueError(f'Image shape {img.shape} is not supported')
    return img


def resize_chw_tensor(img:torch.Tensor, resize_dims: Tuple, mode: str = "bicubic") -> torch.Tensor:
    img = img.unsqueeze(0)                  # [1, c, h, w]
    img = F.interpolate(img, size=resize_dims, mode=mode)
    return img.squeeze(0)                   # [c, h, w]


def min_max_normalize(img:torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    min_value = img.amin(dim=(-2, -1), keepdim=True)
    max_value = img.amax(dim=(-2, -1), keepdim=True)
    return (img - min_value) / (max_value - min_value + eps)


def grayscale_perceptual_features(image: Image,
                                  resize_dims: Tuple[int, int],
                                  resize_mode: str = "bicubic"
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PFM feature extraction used in IUS measure
    :param image:           Grayscale input image for decomposition
    :param resize_dims:     spatial dims (h,w) for feature decomposition - output resolution
    :param resize_mode:     method applied for resizing
    :return:                C-F, L-D, BAND-I, BAND-II Perceptual Feature Components
    """

    image = to_chw_tensor(image)               # [c, h, w]
    if image.shape[0] == 3:
        image = Kcolor.rgb_to_grayscale(image.unsqueeze(0))
        image = image.squeeze(0)               # [c, h, w]

    image = resize_chw_tensor(image, resize_dims=resize_dims, mode=resize_mode)
    image = min_max_normalize(image)

    # coarse - fine
    coarse_fine = Kfilters.sobel(image.unsqueeze(0))
    coarse_fine = coarse_fine.squeeze(0)
    coarse_fine = min_max_normalize(coarse_fine)

    # light - dark
    light_dark = Kfilters.gaussian_blur2d(image.unsqueeze(0),
                                          kernel_size=(7, 7),
                                          sigma=(3.0, 3.0),
                                          border_type="reflect")
    light_dark = light_dark.squeeze(0)
    light_dark = min_max_normalize(light_dark)

    # band-I & band-II
    band_one_mask = ((image >= 0.0) & (image < 0.3)).float()
    band_two_mask = ((image >= 0.3) & (image < 0.6)).float()
    # band_three_mask = ((image >= 0.6) & (image < 0.9)).float()

    band_one = image * band_one_mask
    band_two = image * band_two_mask

    return coarse_fine, light_dark, band_one, band_two


def rgb_perceptual_features(image: Image,
                            resize_dims: Tuple[int, int],
                            resize_mode: str = "bicubic"
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :param image:         RGB input image for decomposition
    :param resize_mode:   method applied for resizing
    :param resize_dims:   spatial dims (h,w) for feature decomposition - output resolution
    :return:              R-G, B-Y, C-F, L-D Perceptual Feature Components
    """

    image = to_chw_tensor(image)                                # [c, h, w]
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    image = resize_chw_tensor(image, resize_dims=resize_dims, mode=resize_mode)
    image = min_max_normalize(image)

    # red-green & blue-yellow
    lab_space_repr = Kcolor.rgb_to_lab(image.unsqueeze(0))       # [1, 3, h, w]
    red_green = lab_space_repr[0, 1:2, :, :]                             # a-channel
    blue_yellow = lab_space_repr[0, 2:3, :, :]                           # b-channel

    red_green = min_max_normalize(red_green)
    blue_yellow = min_max_normalize(blue_yellow)

    # coarse - fine
    grayscale = Kcolor.rgb_to_grayscale(image.unsqueeze(0))     # [1, c, h, w]
    coarse_fine = Kfilters.sobel(grayscale)
    coarse_fine = coarse_fine.squeeze(0)
    coarse_fine = min_max_normalize(coarse_fine)

    # light-dark
    light_dark = Kfilters.gaussian_blur2d(grayscale,
                                          kernel_size=(7, 7),
                                          sigma=(3.0, 3.0),
                                          border_type="reflect")
    light_dark = light_dark.squeeze(0)
    light_dark = min_max_normalize(light_dark)

    return red_green, blue_yellow, coarse_fine, light_dark


class PerceptualFeatureMapTransform:
    def __init__(self,
                 resize_dims: Tuple[int, int],
                 resize_mode: str = "bicubic",
                 data_mode: str = "rgb",):

        data_mode = data_mode.lower()
        assert data_mode in ["rgb", "gray"], " data_mode must be either 'rgb' or 'gray'"

        self.data_mode = data_mode
        self.resize_dims = resize_dims
        self.resize_mode = resize_mode

    def __call__(self, image: Image) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pfms = None
        if self.data_mode == "rgb":
            pfms = rgb_perceptual_features(image=image,
                                           resize_dims=self.resize_dims,
                                           resize_mode=self.resize_mode)
        elif self.data_mode == "gray":
            pfms = grayscale_perceptual_features(image=image,
                                                 resize_dims=self.resize_dims,
                                                 resize_mode=self.resize_mode)

        pfms = torch.stack(pfms, dim=0)     # [4, ch, h, w] . batched from dataloader [bs, 4, ch, h, w]
        return pfms


if __name__ == "__main__":
    import PIL.Image as Image
    image = Image.open('../datasets/kvasir-capsule/test/erosion/5e59c7fdb16c4228_32325.jpg')
    pfms = rgb_perceptual_features(image=image, resize_dims=(128, 128), resize_mode="bicubic")
    collage = torch.cat(pfms, dim=-1).squeeze(0).cpu().numpy()
    collage *= 255
    Image.fromarray(collage.astype(np.uint8)).show()

