import os
import random

import numpy as np
import torch
from PIL import Image


def crop_img(image: np.ndarray, base: int = 64) -> np.ndarray:
    h, w = image.shape[0], image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[
        crop_h // 2:h - crop_h + crop_h // 2,
        crop_w // 2:w - crop_w + crop_w // 2,
        :,
    ]


def crop_patch(im: np.ndarray, pch_size: int) -> np.ndarray:
    h, w = im.shape[0], im.shape[1]
    ind_h = random.randint(0, h - pch_size)
    ind_w = random.randint(0, w - pch_size)
    return im[ind_h:ind_h + pch_size, ind_w:ind_w + pch_size]


def data_augmentation(image: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        out = image.numpy() if hasattr(image, "numpy") else image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image)
    elif mode == 3:
        out = np.flipud(np.rot90(image))
    elif mode == 4:
        out = np.rot90(image, k=2)
    elif mode == 5:
        out = np.flipud(np.rot90(image, k=2))
    elif mode == 6:
        out = np.rot90(image, k=3)
    elif mode == 7:
        out = np.flipud(np.rot90(image, k=3))
    else:
        raise ValueError(f"Invalid image transformation mode: {mode}")
    return out


def random_augmentation(*args):
    flag_aug = random.randint(1, 7)
    return [data_augmentation(d, flag_aug).copy() for d in args]


def np_to_torch(img_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var: torch.Tensor) -> np.ndarray:
    return img_var.detach().cpu().numpy()


def np_to_pil(img_np: np.ndarray) -> Image.Image:
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)
    return Image.fromarray(ar)


def save_image_tensor(image_tensor: torch.Tensor, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    image_np = torch_to_np(image_tensor)[0]
    np_to_pil(image_np).save(output_path)
