# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#=================================================================
#      Data setup
#=================================================================

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


f, a = plt.subplots(1, 1, figsize=(1, 1))
a.imshow(np.reshape(mnist.test.images[1], (28, 28)))
f.show()
plt.draw()
plt.waitforbuttonpress()
