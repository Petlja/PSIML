import numpy as np
from math import sqrt, ceil
from matplotlib import pyplot as plt

def visualize_grid(images, gap = 1):
    """
    Visualizes a grid of images.
    """
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

def visualize_dataset_sample(images, labels, sample_size):
    """
    Selects and visualizes a class-balanced random sample from a given set of labeled images.
    """
    num_classes = len(np.unique(labels))
    sample = images[[i for c in range(num_classes)
                     for i in np.random.permutation(np.flatnonzero(labels == c))[:sample_size]]]
    plt.figure(figsize = (12, 12))
    plt.imshow(visualize_grid(sample))
    plt.axis("off")
    plt.show()

def visualize_classification(image, label, guess_class, guess_prob, class_names):
    """
    Visualizes an image and networks predictions for it.
    """
    print("Ground truth: %s." % (class_names[label],))
    print("Predictions:")
    for cls, prob in zip(guess_class, guess_prob):
        print("Class \"%s\" with probability %.2f%%." % (class_names[cls], prob * 100))
    plt.imshow(image / 255.0)
    plt.axis("off")
    plt.show()
