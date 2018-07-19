"""
Gzip is used for compression.
"""
import gzip
import pickle
from enum import Enum
import os
import sys
import numpy

# <codecell>

def load_data(dataset_path):
    """
    Loads the dataset from given file and returns three subsets: training, validation and test (i.e. blind).

    :param dataset_path: Path to dataset file.
    """
    file = gzip.open(dataset_path, 'rb')
    train_set, valid_set, test_set = pickle.load(file, encoding='latin1')
    file.close()
    return train_set, valid_set, test_set

class ActivationType(Enum):
    """ Defines available activation functions. """
    IDENTITY = 1
    TANH = 2

class IdentityActivation:
    """ Class that implements identity activation function (y = x). """
    @classmethod
    def forward(cls, preactivation):
        """ Implements forward pass. Since identity just returns preactivation as is. """
        # TODO: Replace dummy implementation below with identity forward pass.
        return numpy.zeros(preactivation.shape, dtype=numpy.float)

    @classmethod
    def backward(cls, activation, output_grad):
        """ Implements backward pass. Since identity activation derivative is array of 1s (y = x => dy/dx = 1). """
        # TODO: Replace dummy implementation below with identity backward pass.
        return numpy.zeros(activation.shape, dtype=numpy.float)

class TanhActivation:
    """ Class that implements tanh activation function (y = tanh(x)). """
    @classmethod
    def forward(cls, preactivation):
        """ Implements forward pass. Apply tanh to preactivation and return it. """
        # TODO: Replace dummy implementation below with tanh forward pass.
        return numpy.zeros(preactivation.shape, dtype=numpy.float)

    @classmethod
    def backward(cls, activation, output_grad):
        """ Implements backward pass (y = tanh(x) => dy/dx = (1 - tanh^2(x)). """
        # TODO: Replace dummy implementation below with tanh backward pass.
        return numpy.zeros(activation.shape, dtype=numpy.float)

class SoftmaxWithCrossEntropyLayer:
    """ Class that implements softmax + cross-entropy functionality. """
    def __init__(self, inputs_count):
        """
        Constructor, creates internal objects.

        :param inputs_count: number of inputs (== number of outputs).
        """
        # Outputs of the layer.
        self.y_output = numpy.zeros((inputs_count, 1), dtype=numpy.float)
        # Gradients with respect to inputs.
        self.input_gradients = numpy.zeros((inputs_count, 1), dtype=numpy.float)
        # The most probable class (index of max output).
        self.y_max = None

    def forward(self, x_input):
        """
        Performs forward pass on this layer.

        :param x_input: Input array for this layer.
        """
        # Calculate output as softmax of inputs.
        # TODO: Implement softmax below as [y_output] = softmax([x_input]).
        # Save the most probable class.
        self.y_max = numpy.argmax(self.y_output)

    def backward(self, target):
        """
        Performs backward pass on this layer.

        :param target: Expected output (to be used in loss function).
        """
        # Gradients are calculated using [gradient] = [output] - [target].
        # TODO: Implement softmax + crossentropy backward compute below as [gradient] = [output] - [target].

    def prediction(self):
        """ Returns the most probably class for the most recent forward call. """
        return self.y_max


class FullyConnectedLayer:
    """ Class that implements fully connected layer functionality. """
    def __init__(self, rng, inputs_count, outputs_count, activation_type):
        """
        Constructor, creates internal objects.

        :param rng: A random number generator used to initialize weights.
        :param inputs_count: Dimensionality of input.
        :param outputs_count: Dimensionality of output.
        """
        # Create weights array and initialize it randomly.
        self.w_weights = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (inputs_count + outputs_count)),
                high=numpy.sqrt(6. / (inputs_count + outputs_count)),
                size=(outputs_count, inputs_count)
            ),
            dtype=numpy.float
        )
        # Create biases and zero them out.
        self.b_biases = numpy.zeros((outputs_count, 1), dtype=numpy.float)
        # Allocate all gradient arrays.
        self.weight_gradients = numpy.zeros((outputs_count, inputs_count), dtype=numpy.float)
        self.bias_gradients = numpy.zeros((outputs_count, 1), dtype=numpy.float)
        self.input_gradients = numpy.zeros((inputs_count, 1), dtype=numpy.float)
        # Set activation function according to given type.
        if activation_type == ActivationType.IDENTITY:
            self.activation = IdentityActivation()
        else:
            self.activation = TanhActivation()
        # Declare input/output arrays to be used later.
        self.x_input = None
        self.y_output = numpy.zeros((outputs_count, 1), dtype=numpy.float)

    def forward(self, x_input):
        """
        Performs forward pass for this layer using formula [output] = activation([input] * [wights] + [biases])

        :param x_input: Input array for forward pass.
        """
        # TODO: Implement fully connected layer forward compute below as
        # [output] = activation([weights] * [input] + [bias]).
        # Remember input to be able to compute gradients with respect to weights.
        self.x_input = x_input

    def backward(self, output_grad):
        """
        Performs backward pass for this layer.

        :param output_grad: Gradients of the outputs.
        """
        # Calculate preactivation gradients.
        pre_activation_gradient = self.activation.backward(self.y_output, output_grad)
        # Based on preactivation gradient calculate bias, weights and input gradients.
        # TODO: Implement fully connected layer backward. Compute bias_gradients, weight_gradients and input_gradients.

    def update_weights(self, alpha):
        """
        Updates weights for this layer using given learning rate and already computed gradients.

        :param alpha: Learning rate to be used.
        """
        self.w_weights -= alpha * self.weight_gradients
        self.b_biases -= alpha * self.bias_gradients


class MLPNetwork:
    """
    Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model that has one hidden fully connected layer,
    and one output fully connected layer, both with nonlinear activations. The top layer (third one) is a softmax layer.
    """
    def __init__(self, rng, inputs_count, hidden_count, outputs_count):
        """Initialize the parameters for the multilayer perceptron.

        :param rng: A random number generator used to initialize weights
        :param inputs_count: Number of input units, the dimension of the space in which the datapoints lie.
        :param hidden_count: number of hidden units.
        :param outputs_count: Number of output units, the dimension of the space in which the labels lie.
        """
        # Create hidden layer (first fully connected layer).
        self.hidden_layer = FullyConnectedLayer(
            rng=rng,
            inputs_count=inputs_count,
            outputs_count=hidden_count,
            activation_type=ActivationType.TANH
        )

        # Create output layer (second fully connected layer).
        self.output_layer = FullyConnectedLayer(
            rng=rng,
            inputs_count=hidden_count,
            outputs_count=outputs_count,
            activation_type=ActivationType.IDENTITY
        )

        # Create softmax layer.
        self.softmax_layer = SoftmaxWithCrossEntropyLayer(inputs_count=outputs_count)

    def forward(self, x_input):
        """ Performs forward pass through the network by sequentially invoking forward on child layers.

        :param x_input: Input to the network.
        """
        self.hidden_layer.forward(x_input)
        self.output_layer.forward(self.hidden_layer.y_output)
        self.softmax_layer.forward(self.output_layer.y_output)

    def backward(self, t_target):
        """ Performs backward pass through the network by sequentially invoking backward on child layers in reverse
        order.

        :param t_target: Expected output of the network.
        """
        self.softmax_layer.backward(t_target)
        self.output_layer.backward(self.softmax_layer.input_gradients)
        self.hidden_layer.backward(self.output_layer.input_gradients)

    def update_weights(self, alpha):
        """ Performs update of the network weights by updating weight on each child layer.

        :param alpha: Learning rate to be used for weight update.
        """
        self.output_layer.update_weights(alpha)
        self.hidden_layer.update_weights(alpha)

    def test(self, dataset):
        """ Performs testing of the trained network on the given dataset. Returns accuracy of the network.

        :param dataset: Dataset to be used for testing.
        """
        x_input, y_target = dataset
        error_count = 0
        for i in range(x_input.shape[0]):
            self.forward(x_input[i].reshape(x_input.shape[1], 1))
            if self.softmax_layer.prediction() != y_target[i]:
                error_count += 1
        return float(error_count) / x_input.shape[0]


def train_nn_with_sgd(dataset_path, epochs_count, alpha):
    """ Performs training of the MLP network on the given dataset using stochastic gradient descent.

        :param dataset_path: Path to dataset file.
        :param epochs_count: Number of epochs to run training.
        :param alpha: Learning rate to be used during training.
    """
    # Load the datasets.
    train_set, valid_set, test_set = load_data(dataset_path)
    # Initialize neural network.
    neural_net = MLPNetwork(numpy.random, 28 * 28, 100, 10)
    # Print header.
    print('Epoch\tTrainingError%%\tValidationError%%\tTestError%%')
    # Train network for limited number of epochs.
    train_input, train_target = train_set
    for epoch in range(epochs_count):
        # Go over all samples from test set (shape of input is 50000 x 784 since we have 50000 images of a size 784
        # (784 == 28 x 28)).
        for i in range(train_input.shape[0]):
            # Take current example.
            train_input_reshaped = train_input[i].reshape(train_input.shape[1], 1)
            # Perform forward pass on current example.
            neural_net.forward(train_input_reshaped)
            # Back-propagate error.
            neural_net.backward(train_target[i])
            # Use gradients from back-propagation to update weights.
            neural_net.update_weights(alpha)
        # Measure and print accuracy on all data sets.
        train_error = neural_net.test(train_set)
        valid_error = neural_net.test(valid_set)
        test_error = neural_net.test(test_set)
        print('%d\t%f\t%f\t%f' %(epoch, 100 * train_error, 100 * valid_error, 100 * test_error))
    # Save the trained network.
    gzip_file_path_os_normalized = os.path.join(".","nn.pkl.gz")
    gzip_file = gzip.open(gzip_file_path_os_normalized, 'wb')
    pickle.dump(neural_net, gzip_file)
    gzip_file.close()

def test_nn(nn_path, dataset_path):
    """ Performs testing of the MLP network on the given dataset.

        :param nn_path: Path to nn file.
        :param dataset_path: Path to dataset file.
    """
    # Load neural network.
    nn_file = gzip.open(nn_path, 'rb')
    neural_net = pickle.load(nn_file, encoding='latin1')
    nn_file.close()
    # Load datasets.
    ds_file = gzip.open(dataset_path, 'rb')
    _, _, test_set = pickle.load(ds_file, encoding='latin1')
    ds_file.close()
    # Test.
    print('Test neural net %s on test set in %s' % (nn_path, dataset_path))
    test_error = neural_net.test(test_set)
    print('Test error: %f' %(100 * test_error))

class Scenario(Enum):
    """ Defines all possible scenarios. """
    TEST = 1
    TRAIN = 2

# <codecell>

if __name__ == "__main__":
    # Determine the scenario to run.
    SCENARIO = Scenario.TEST
    if len(sys.argv) != 1:
        assert (len(sys.argv) == 2), "None or exactly one (\"train\" or \"test\") argument must be provided."
        if sys.argv[1] == "test":
            SCENARIO = Scenario.TEST
        elif sys.argv[1] == "train":
            SCENARIO = Scenario.TRAIN
        else:
            raise ValueError("Argument value must be \"train\" or \"test\".")

    dataset_path_os_normalized = os.path.join(".","data","mnist.pkl.gz")
    model_path_os_normalized = os.path.join(".","model","nn.pkl.gz")
    if SCENARIO == Scenario.TRAIN:
        train_nn_with_sgd(dataset_path=dataset_path_os_normalized, epochs_count=10, alpha=0.01)
    else:
        test_nn(model_path_os_normalized, dataset_path_os_normalized)
