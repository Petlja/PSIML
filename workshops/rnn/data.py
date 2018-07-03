import numpy as np


class DataProvider:
    def __init__(self, examples, sequence_length, inputs, outputs):
        np.random.seed(1) # Set the seed explicitly so that the same data is generated every time.
        self._x = np.random.randn(examples, sequence_length, inputs)  # random values of input length
        self._z = np.zeros((examples, sequence_length, outputs))
        self._z[:, 2::3, 0] = 1  # learn to count to three regardless of input x

    def get_example_count(self):
        return self._x.shape[0]

    def get_example(self, idx):
        x = self._x[idx, :, :]
        z = self._z[idx, :, :]
        return x, z
