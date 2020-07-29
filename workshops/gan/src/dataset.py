import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FacesDataSet(Dataset):
    def __init__(self, img_size, crop_size=128, total_images=None):
        assert img_size <= crop_size <= 250
        self.img_size = img_size
        self.crop_size = crop_size
        project_path = os.path.dirname(os.path.dirname(__file__)) # should be PSIML/workshops/gan
        project_path = os.path.abspath(project_path)
        self.data_root = os.path.join(project_path, "data", "lfw-deepfunneled")
        self.image_paths = glob.glob(os.path.join(self.data_root, "**", "*.jpg"))
        self.image_paths = self.image_paths[:total_images]
        self.transforms = transforms.Compose(
            [
                transforms.CenterCrop((self.crop_size, self.crop_size)),
                transforms.Resize(self.img_size),
                transforms.ToTensor() # this puts the image in the CxHxW format and normalizes it to [0,1). See if we want this!
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path =  self.image_paths[index]
        image = Image.open(image_path)
        return self.transforms(image)
