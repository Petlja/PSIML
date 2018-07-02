import numpy as np
import os
from math import sqrt, ceil
import pickle

def load_pickle(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file, encoding='latin1')

def load_cifar10_batches(file_names):
    dicts = [load_pickle(fn) for fn in file_names]
    x = np.concatenate([d['data'] for d in dicts]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
    y = np.concatenate([np.asarray(d['labels']) for d in dicts])
    return x, y

def load_cifar10(root):
    """Load CIFAR-10 dataset."""

    x_dev, y_dev = load_cifar10_batches([os.path.join(root, 'data_batch_%d' % b) for b in range(1, 6)])
    x_test, y_test = load_cifar10_batches([os.path.join(root, 'test_batch'),])
    return x_dev, y_dev, x_test, y_test

def visualize_grid(images, gap = 1):
    """ Visualize a grid of images """
    (N, H, W, C) = images.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A * (H + gap) - gap, A * (W + gap) - gap, C), images.dtype)
    G *= np.min(images)
    for n in range(N):
        y = n // A
        x = n % A
        offset_h = y * (H + gap)
        offset_w = x * (H + gap)
        G[offset_h : offset_h + H, offset_w : offset_w + W, :] = images[n, :, :, :]
    # Normalize to [0,1].
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G
