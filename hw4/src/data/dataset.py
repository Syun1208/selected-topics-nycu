import os
import random
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from src.utils.io_utils import crop_img, random_augmentation


class PromptTrainDataset(Dataset):

    DE_DICT = {"derain": 0, "desnow": 1}

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rs_ids: List[dict] = []
        self.snow_ids: List[dict] = []
        self.de_type = list(args.de_type)

        self._init_ids()
        self._merge_ids()
        self.toTensor = ToTensor()

    def _init_ids(self):
        if "derain" in self.de_type:
            self._init_rs_ids()
        if "desnow" in self.de_type:
            self._init_snow_ids()
        random.shuffle(self.de_type)

    def _init_rs_ids(self):
        rs_list = os.path.join(
            os.path.dirname(self.args.data_file_dir.rstrip("/")), "rain.txt",
        )
        with open(rs_list) as f:
            temp_ids = [
                self.args.derain_dir + line.strip()
                for line in f if line.strip()
            ]
        self.rs_ids = [
            {"clean_id": x, "de_type": 0} for x in temp_ids
        ] * self.args.num_aug
        print(f"Total Rainy Ids : {len(self.rs_ids)}")

    def _init_snow_ids(self):
        snow_list = os.path.join(
            os.path.dirname(self.args.data_file_dir.rstrip("/")), "snow.txt",
        )
        with open(snow_list) as f:
            temp_ids = [
                self.args.desnow_dir + line.strip()
                for line in f if line.strip()
            ]
        self.snow_ids = [
            {"clean_id": x, "de_type": 1} for x in temp_ids
        ] * self.args.num_aug
        print(f"Total Snow Ids : {len(self.snow_ids)}")

    def _crop_patch(self, img_1, img_2):
        h, w = img_1.shape[:2]
        ind_h = random.randint(0, h - self.args.patch_size)
        ind_w = random.randint(0, w - self.args.patch_size)
        patch_1 = img_1[
            ind_h:ind_h + self.args.patch_size,
            ind_w:ind_w + self.args.patch_size,
        ]
        patch_2 = img_2[
            ind_h:ind_h + self.args.patch_size,
            ind_w:ind_w + self.args.patch_size,
        ]
        return patch_1, patch_2

    @staticmethod
    def _get_gt_name(degraded_name: str) -> str:
        return degraded_name.replace("degraded", "clean").replace("-", "_clean-")

    def _merge_ids(self):
        self.sample_ids = []
        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids
        if "desnow" in self.de_type:
            self.sample_ids += self.snow_ids
        print(f"Total sample ids: {len(self.sample_ids)}")

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        degrad_img = crop_img(
            np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16,
        )
        clean_name = self._get_gt_name(sample["clean_id"])
        clean_img = crop_img(
            np.array(Image.open(clean_name).convert("RGB")), base=16,
        )

        degrad_patch, clean_patch = random_augmentation(
            *self._crop_patch(degrad_img, clean_img)
        )

        return (
            [clean_name, de_id],
            self.toTensor(degrad_patch),
            self.toTensor(clean_patch),
        )

    def __len__(self):
        return len(self.sample_ids)


class TestSpecificDataset(Dataset):

    EXTENSIONS = ("jpg", "JPG", "png", "PNG", "jpeg", "JPEG", "bmp", "BMP")

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.degraded_ids: List[str] = []
        self._init_clean_ids(args.test_path)
        self.toTensor = ToTensor()

    def _init_clean_ids(self, root: str):
        if os.path.isdir(root):
            name_list = sorted(
                f for f in os.listdir(root)
                if any(f.endswith(ext) for ext in self.EXTENSIONS)
            )
            if not name_list:
                raise FileNotFoundError(f"No image files found in {root}")
            self.degraded_ids = [root + name for name in name_list]
        else:
            if not any(root.endswith(ext) for ext in self.EXTENSIONS):
                raise FileNotFoundError(f"Not an image file: {root}")
            self.degraded_ids = [root]
        print(f"Total Images : {len(self.degraded_ids)}")
        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(
            np.array(Image.open(self.degraded_ids[idx]).convert("RGB")), base=16,
        )
        name = self.degraded_ids[idx].split("/")[-1][:-4]
        return [name], self.toTensor(degraded_img)

    def __len__(self):
        return self.num_img


def resolve_test_path(path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        degraded = os.path.join(path, "degraded")
        if os.path.isdir(degraded):
            path = degraded
        if not path.endswith("/"):
            path = path + "/"
    return path
