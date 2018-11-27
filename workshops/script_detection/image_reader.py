import random
import utils.io
import glob
import os
from PIL import Image
import numpy as np


def _load_image(image_path, height):
    image = Image.open(image_path).convert('L')  # greyscale
    width_orig, height_orig = image.size
    width = int(width_orig * height / height_orig)
    image = image.resize(size=(width, height))
    data = np.asarray(image, dtype="int32")
    assert len(data.shape) == 2
    return data


def _get_pngs(dir):
    return glob.glob(os.path.join(dir, "*.png"))


def _read_data(root_dir):
    images = []
    labels = []
    class_dirs = utils.io.get_subdirs(root_dir)
    for label, class_dir in enumerate(class_dirs):
        for image in _get_pngs(class_dir):
            images.append(image)
            labels.append(label)
    return images, labels


def _norm_image(image):
    image = image / 255  # 1 - white, 0 - black
    image = image * (-1) + 1  # 0 - white, 1 - black
    return image


def _read_image(image_path, line_height):
    image = _load_image(image_path, height=line_height)
    image = _norm_image(image)
    return image


class ImageReader(object):
    def __init__(self, line_height, root_dir):
        self.line_height = line_height
        self._images, self._labels = _read_data(root_dir)
        self._labels = [np.array([label]) for label in self._labels]
        self._images = [_read_image(image, line_height) for image in self._images]
        self._images = [image.T for image in self._images]  # transpose to have width x height
        self._indices = list(range(self.__len__()))

    def shuffle(self):
        random.shuffle(self._indices)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        index = self._indices[index]
        image = self._images[index]
        label = self._labels[index]
        return image, label
