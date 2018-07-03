from builtins import range
from math import sqrt, ceil
import numpy as np

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