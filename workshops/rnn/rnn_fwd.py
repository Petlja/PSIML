import numpy as np
import matplotlib.pyplot as plt
from data import DataProvider


# Implementation follows "Supervised Sequence Labelling with Recurrent Neural Networks", by Alex Graves
# Book is available at https://www.cs.toronto.edu/~graves/preprint.pdf


def plot_predictions(y):
    t = np.array(list(y.keys()))
    predictions = np.array(list(y.values()))
    output_size = predictions.shape[-1]
    plt.clf()
    for o in range(output_size):
        plt.plot(t, predictions[:, o])
    plt.show()


def save_params(file_path, W_ih, b_ih, W_hh, W_hk, b_hk):
    np.savez(file_path, W_ih=W_ih, b_ih=b_ih, W_hh=W_hh, W_hk=W_hk, b_hk=b_hk)


def load_params(file_path):
    data = np.load(file_path)
    return data['W_ih'], data['b_ih'], data['W_hh'], data['W_hk'], data['b_hk']


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))     # (3.4)


# Vanilla RNN.
def main():
    hidden_units = 15
    outputs = 1
    inputs = 5
    examples = 100
    sequence_length = 10
    learning_rate = 0.001

    # Shapes of weights:
    # Input -> hidden connections
    # W_ih.shape = (inputs, hidden_units)
    # b_ih.shape = (hidden_units,)
    # Hidden -> hidden connections
    # W_hh.shape = (hidden_units, hidden_units)
    # Hidden -> output connections
    # W_hk.shape = (hidden_units, outputs)
    # b_hk.shape = (outputs,)

    # Load trained network.
    filename = r"model/trained_net.wts.npz"
    W_ih, b_ih, W_hh, W_hk, b_hk = load_params(filename)

    # Get training set.
    d = DataProvider(examples, sequence_length, inputs, outputs)
    # x.shape = (sequence_length, inputs)
    # z.shape = (sequence_length, 1)
    x, z = d.get_example(0)

    # dictionary where key is the timestamp.
    a_h = {}
    b_h = dict()
    b_h[-1] = np.zeros_like(b_ih)
    a_k = {}
    y = {}
    for t in range(sequence_length):
        a_h[t] = None  # TODO: (3.30), don't forget bias parameter
        b_h[t] = None  # TODO: (3.31), hint: theta_h = tanh
        a_k[t] = None  # TODO: (3.32), don't forget bias parameter
        y[t] = None  # TODO: Binary classification
    plot_predictions(y)


if __name__ == "__main__":
    main()
