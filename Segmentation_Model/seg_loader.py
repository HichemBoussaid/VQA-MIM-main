import os.path
import json
import random

import numpy as np
import torch
import tqdm
from skimage import io
from skimage import morphology
from torch.utils.data import Dataset

RANDOM_SEED = 42


class SegLoader(Dataset):
    def __init__(self, imgFolder, images_file, transform=None, train=True, ratio_images_to_use=1.0, patch_size=1000):
        with open(images_file) as json_data:
            self.imagesJSON = json.load(json_data)

        self.images = [img["id"] for img in self.imagesJSON['images'] if img['active'] ]

        self.images = self.images[:int(len(self.images) * ratio_images_to_use)]
        self.imgFolder = imgFolder
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        dept, folder = image_id.split("-")
        image_name = \
            [x for x in os.listdir(os.path.join(self.imgFolder, dept, folder)) if
             "BDOrtho_" in x and "aux.xml" not in x][0]
        seg_image_name = [x for x in os.listdir(os.path.join(self.imgFolder, dept, folder, "rasters")) if
                          "raster" in x and ".json" not in x][0]
        img = io.imread(os.path.join(self.imgFolder, dept, folder, image_name))
        seg_img = io.imread(os.path.join(self.imgFolder, dept, folder, "rasters", seg_image_name))
        indexes = seg_img != 0
        seg_img[indexes] = 1
        seg_img = seg_img.transpose(2, 0, 1)
        seg_img = seg_img.astype(np.uint8)
        

        """if seg_img.shape[0] <16:
            return None"""
        img = img.astype(np.uint8)
        disk = morphology.disk(11)
        seg_img[11] = morphology.dilation(seg_img[11], disk)
        seg_img[12] = morphology.dilation(seg_img[12], disk)
        if self.transform:
            img = self.transform(img.copy())

        return img, seg_img
