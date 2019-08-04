import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow import keras

_FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_fashion_mnist():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
    # normalize images
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, train_labels, test_images, test_labels


def get_fashion_mnist_class_name(class_id):
    return _FASHION_MNIST_CLASSES[class_id]


def show_random_images(images, labels, count):
    np.random.seed(10)
    rnd_ind = np.random.choice(len(images), count)
    for i, ind in enumerate(rnd_ind):
        if i % 25 == 0:
            if i > 0:
                plt.show()
            plt.figure(figsize=(8, 8))
        plt.subplot(5, 5, (i % 25)+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[ind], cmap=plt.cm.binary)
        plt.xlabel(_FASHION_MNIST_CLASSES[labels[ind]])
    plt.show()
