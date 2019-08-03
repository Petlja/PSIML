import numpy as np
import math
import random
import matplotlib.pyplot as plt
from itertools import chain


def euclidian(ax, ay, bx, by):
    return math.sqrt((ax-bx)**2 + (ay-by)**2)


def generate_data_circle(numSamples, noise=0.1):
    random.seed(2019)
    points = []
    radius = 5

    def get_label(point_x, point_y, center_x, center_y):
        return 1 if (euclidian(point_x, point_y, center_x, center_y) < (radius * 0.5)) else 0

    # Generate positive points inside the circle.
    for _ in range(numSamples // 2):
        r = random.uniform(0.0, radius * 0.5)
        angle = random.uniform(0.0, 2.0 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        noiseX = random.uniform(-radius, radius) * noise
        noiseY = random.uniform(-radius, radius) * noise
        label = get_label(x + noiseX, y + noiseY, 0, 0)
        points.append([x, y, label])

    # Generate negative points outside the circle.
    for _ in range(numSamples // 2):
        r = random.uniform(radius * 0.7, radius)
        angle = random.uniform(0.0, 2.0 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        noiseX = random.uniform(-radius, radius) * noise
        noiseY = random.uniform(-radius, radius) * noise
        label = get_label(x + noiseX, y + noiseY, 0, 0)
        points.append([x, y, label])

    random.shuffle(points)
    return np.array(points)

def generate_data_spiral(numSamples, noise=0.1):
    random.seed(2019)
    radius = 5

    def generate_one_spiral(deltaT, label):
        for i in range(numSamples):
            r = i / numSamples * radius
            t = 1.75 * i / numSamples * 2 * math.pi + deltaT
            x = r * math.sin(t) + random.uniform(-1, 1) * noise
            y = r * math.cos(t) + random.uniform(-1, 1) * noise
            yield [x, y, label]

    points = list(chain(generate_one_spiral(0, 1), generate_one_spiral(math.pi, 0)))

    random.shuffle(points)
    return np.array(points)


def generate_data_xor(numSamples, noise=0.1):
    random.seed(2019)
    radius = 5
    padding = 0.3

    def get_label(p):
        return 1 if p[0] * p[1] >= 0 else 0

    points = []
    for i in range(numSamples):
        x = random.uniform(-radius, radius)
        x += padding * (1 if x > 0 else -1)
        y = random.uniform(-radius, radius)
        y += padding * (1 if y > 0 else -1)
        noiseX = random.uniform(-radius, radius) * noise
        noiseY = random.uniform(-radius, radius) * noise
        label = get_label([x + noiseX, y + noiseY])
        points.append([x, y, label])

    random.shuffle(points)
    return np.array(points)

def generate_data_gauss(numSamples, noise=0.1):
    random.seed(2019)
    radius = 5

    def generate_one_gauss(cx, cy, label):
        for i in range(numSamples):
            noiseX = random.uniform(-radius, radius) * noise
            noiseY = random.uniform(-radius, radius) * noise
            x = random.gauss(cx, 1) + noiseX
            y = random.gauss(cy, 1) + noiseY
            yield [x, y, label]

    points = list(chain(generate_one_gauss(radius // 4, radius // 4, 1), generate_one_gauss(-radius // 4, -radius // 4, 0)))

    random.shuffle(points)
    return np.array(points)


def split_data(data, val_factor=0.3):
    random.shuffle(data)
    split = int(len(data) * val_factor)
    return np.array(data[split:]), np.array(data[:split])


def plot_data(train_data=None, val_data=None, show=False):
    # Dimensions of a grid to plot
    x_min, x_max = -6, 6
    y_min, y_max = -6, 6
    fig = plt.figure(figsize=(6, 6))

    # Plot points
    if train_data is not None:
        # Different colors for labels
        label_colors_train = ['b' if point[2] ==
                              0 else 'y' for point in train_data]
        plt.scatter(train_data[:, 0], train_data[:, 1],
                    c=label_colors_train, edgecolors='w', s=40)
    if val_data is not None:
        # Different colors for labels
        label_colors_val = ['b' if point[2] ==
                            0 else 'y' for point in val_data]
        plt.scatter(val_data[:, 0], val_data[:, 1],
                    c=label_colors_val, edgecolors='k', s=20)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    if show:
        plt.show()
    return fig
