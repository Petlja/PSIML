import numpy as np
import matplotlib.pyplot as plt
from data import DataProvider


# Implementation follows "Supervised Sequence Labelling with Recurrent Neural Networks", by Alex Graves
# Book is available at https://www.cs.toronto.edu/~graves/preprint.pdf


def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))  # (3.13)


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


def d_tanh(x):
    return 1 - np.tanh(x) * np.tanh(x)  # (3.6)


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))     # (3.4)


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)      # (3.7)


def get_loss(y, z):
    # cross-entropy, two versions:
    # 1) -sum(expected * log predicted)
    # 2) -sum(expected * log predicted + (1-expected) log (1-predicted))
    # In case of 1) softmax will handle that only one output gets activated
    # If we don't use softmax, i.e. outputs are not exclusive, we have to use 2).
    # TODO: see Bishop, 95, chapter 6
    if len(y) == 1:
        return (z - 1) * np.log(1 - y) - z * np.log(y)  # (3.15)
    else:
        return -np.sum(z * np.log(y))  # (3.16)


# Vanilla RNN.
if __name__ == "__main__":
    hidden_units = 15
    outputs = 1
    inputs = 5
    examples = 100
    sequence_length = 10
    weight_norm_factor = 0.01
    epochs = 1000
    momentum = 0.9
    learning_rate = 0.001

    np.random.seed(1) # Initialize with the seed so that results are reproducable.

    # Input -> hidden connections
    W_ih = np.random.randn(inputs, hidden_units) * weight_norm_factor
    b_ih = np.zeros((hidden_units,))
    # Hidden -> hidden connections
    W_hh = np.random.randn(hidden_units, hidden_units) * weight_norm_factor
    # Hidden -> output connections
    W_hk = np.random.randn(hidden_units, outputs) * weight_norm_factor
    b_hk = np.zeros((outputs,))

    # Get training set.
    d = DataProvider(examples, sequence_length, inputs, outputs)

    update_W_ih = np.zeros_like(W_ih)
    update_b_ih = np.zeros_like(b_ih)
    update_W_hh = np.zeros_like(W_hh)
    update_W_hk = np.zeros_like(W_hk)
    update_b_hk = np.zeros_like(b_hk)

    loss = dict()
    for epoch in range(epochs):
        loss[epoch] = 0
        print(str.format("Epoch {0} / {1}", epoch, epochs))
        for d_id in range(d.get_example_count()):
            x, z = d.get_example(d_id)
            # Auxiliary variables.
            a_h = {}
            b_h = dict()
            b_h[-1] = np.zeros_like(b_ih)
            a_k = {}
            y = {}
            for t in range(sequence_length):
                a_h[t] = np.dot(x[t], W_ih) + b_ih + np.dot(b_h[t - 1], W_hh)  # (3.30)
                b_h[t] = np.tanh(a_h[t])  # (3.31)
                a_k[t] = np.dot(b_h[t], W_hk) + b_hk  # (3.32)
                y[t] = sigmoid(a_k[t])  # Binary classification
                loss[epoch] += get_loss(y[t], z[t])
                # plot_predictions(y)

            # Back-propagation through time
            # Calculate deltas
            delta_k = {}
            delta_h = dict()
            delta_h[sequence_length] = np.zeros((hidden_units,))
            for t in reversed(range(sequence_length)):
                delta_k[t] = y[t] - z[t]  # (3.19), (3.23)
                delta_h_hk = np.dot(W_hk, delta_k[t])
                delta_h_hh = np.dot(W_hh, delta_h[t + 1])
                delta_h[t] = d_tanh(a_h[t]) * (delta_h_hk + delta_h_hh)  # (3.33)
            # Calculate gradients, everything is derived from (3.35)
            d_W_ih = np.zeros_like(W_ih)
            d_b_ih = np.zeros_like(b_ih)
            d_W_hh = np.zeros_like(W_hh)
            d_W_hk = np.zeros_like(W_hk)
            d_b_hk = np.zeros_like(b_hk)
            for t in range(sequence_length):
                d_W_ih += np.dot(x[t].reshape(-1, 1), delta_h[t].reshape(-1, 1).T)
                d_b_ih += delta_h[t]  # * 1
                d_W_hh += np.dot(b_h[t].reshape(-1, 1), delta_h[t].reshape(-1, 1).T)
                d_W_hk += np.dot(b_h[t].reshape(-1, 1), delta_k[t].reshape(-1, 1).T)
                d_b_hk += delta_k[t]  # * 1

            # Clip gradients to mitigate exploding gradients
            clip_threshold = 3
            for gradients in [d_W_ih, d_b_ih, d_W_hh, d_W_hk, d_b_hk]:
                np.clip(gradients, -clip_threshold, clip_threshold, out=gradients)

            # Calculate updates using momentum and learning rate, apply (3.39)
            update_W_ih = momentum * update_W_ih - learning_rate * d_W_ih
            update_b_ih = momentum * update_b_ih - learning_rate * d_b_ih
            update_W_hh = momentum * update_W_hh - learning_rate * d_W_hh
            update_W_hk = momentum * update_W_hk - learning_rate * d_W_hk
            update_b_hk = momentum * update_b_hk - learning_rate * d_b_hk
            # Apply gradients
            W_ih += update_W_ih
            b_ih += update_b_ih
            W_hh += update_W_hh
            W_hk += update_W_hk
            b_hk += update_b_hk
    plot_predictions(loss)

    filename = "trained_net.wts.npz"
    save_params(filename, W_ih, b_ih, W_hh, W_hk, b_hk)

    W_ih, b_ih, W_hh, W_hk, b_hk = load_params(filename)
    x, z = d.get_example(0)
    a_h = {}
    b_h = dict()
    b_h[-1] = np.zeros_like(b_ih)
    a_k = {}
    y = {}
    for t in range(sequence_length):
        a_h[t] = np.dot(x[t], W_ih) + b_ih + np.dot(b_h[t - 1], W_hh)  # (3.30)
        b_h[t] = np.tanh(a_h[t])  # (3.31)
        a_k[t] = np.dot(b_h[t], W_hk) + b_hk  # (3.32)
        y[t] = sigmoid(a_k[t])  # Binary classification
    plot_predictions(y)