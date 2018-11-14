import numpy as np
import tensorflow as tf

def image_placeholder(height, width, channels, name):
    """
    Creates an input placeholder to hold a batch of images with known spatial dimensions and number
    of channels. The placeholder will later be bound to actual image data.

    :param height: Height in pixels.
    :param width: Width in pixels.
    :param channels: Number of channels (e.g. 1 for grayscale images, and 3 for RGB images).
    :param name: Name of the returned tensor.

    :returns: Tensor of shape (<batch_size>, height, width, channels), holding 32-bit floating point
        numbers, where <batch_size> is the number of examples in a batch, which is unknown at network
        construction time.
    """

    # Hints:
    # - Use [tf.placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
    # - Shape dimensions with value None are set automatically once actual data is provided.

    return tf.placeholder(dtype=tf.float32, shape=[None, height, width, channels], name=name)

def conv_layer(inputs, filters, kernel_size, strides, with_activation, name):
    """
    Creates a convolutional layer with bias, with or without ReLU activation, and "same" padding.

    :param inputs: Tensor containing inputs of the layer.
    :param filters: Number of convolutional filters.
    :param kernel_size: Kernel size of the convolutional filters, pair of integers if height and width are
        different, or a single number if they are equal.
    :param strides: Strides of the convolutional filters, pair of integers if height and width are
        different, or a single number if they are equal.
    :param with_activation: Flag indicating whether to include ReLU activation.
    :param name: Name of the returned tensor.

    :returns: Tensor containing outputs of the layer.
    """

    # Clarification:
    # "Same" padding: padding ~ kernel_size / 2 (center of the window must stay within input).
    # "Valid" padding: padding = 0 (the whole window must stay within input).

    # Hints:
    # - For convolutional layer use [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d).
    # - For ReLU activation function use [tf.nn.relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu).
    # - Bias is enabled by default for convolutional layers.

    activation = tf.nn.relu if with_activation else None
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="SAME",
        activation=activation,
        name=name)

def pool_layer(inputs, pool_size, strides, name):
    """
    Creates a max-pooling layer with "same" padding.

    :param inputs: Tensor containing inputs of the layer.
    :param pool_size: Size of the pooling window, pair of integers if height and width are different, or
        a single number if they are equal.
    :param strides: Strides of the pooling window, pair of integers if height and width are different, or
        a single number if they are equal.
    :param name: Name of the returned tensor.

    :returns: Tensor containing outputs of the layer.
    """

    # Hint:
    # Use [tf.layers.max_pooling2d](https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling2d).

    return tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=pool_size,
        strides=strides,
        padding="SAME",
        name=name)

def fc_layer(inputs, units, with_activation, name):
    """
    Creates a fully connected layer with bias, with or without ReLU activation.

    :param inputs: Tensor containing inputs of the layer.
    :param units: Number of output units.
    :param with_activation: Flag indicating whether to include ReLU activation.
    :param name: Name of the returned tensor.

    :returns: Tensor containing outputs of the layer.
    """

    # Hints:
    # - For fully connected layer use [tf.layers.dense](https://www.tensorflow.org/api_docs/python/tf/layers/dense).
    # - Bias is enabled by default for fully connected layers.
    # - For ReLU activation function use [tf.nn.relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu).
    # - Input to a fully connected layer needs to be flat (one-dimensional).
    # - One way to flatten is using [tf.reshape](https://www.tensorflow.org/api_docs/python/tf/reshape).
    # - Tensor shape can be accessed using [Tensor.shape](https://www.tensorflow.org/api_docs/python/tf/Tensor#shape).

    inputs_flat = inputs if len(inputs.shape) < 2 else tf.reshape(inputs, [-1, np.prod(inputs.shape[1:])])
    activation = tf.nn.relu if with_activation else None
    return tf.layers.dense(inputs=inputs_flat, units=units, activation=activation, name=name)

def label_placeholder(name):
    """
    Creates an input placeholder to hold a batch of image labels. The placeholder will later be bound
    to actual label data.

    :param name: Name of the returned tensor.

    :returns: Tensor of shape (<batch_size>,), holding 64-bit integers, where <batch_size> is the number
    of examples in a batch, which is unknown at network construction time.
    """

    # Hints:
    # - Use [tf.placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
    # - Shape dimensions with value None are set automatically once actual data is provided.

    return None

def cross_entropy_loss(labels, logits, name):
    """
    Creates a subnetwork that computes cross-entropy loss.

    :param labels: Ground truth labels.
    :param logits: Logits output by the network.
    :param name: Name of the returned tensor.

    :returns: Tensor containing one scalar equal to the sum of cross-entropy values for all
        examples in batch.
    """

    # Clarification:
    # Logits are per-class "scores" output by the network (that turn into probabilities after
    # applying softmax function). When computing cross-entropy loss, it is more numerically
    # stable to merge softmax function with it.

    # Hints:
    # - Use [`tf.nn.sparse_softmax_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits).
    # - Use [`tf.reduce_sum`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum) to sum over examples in batch.

    return None

def classification_accuracy(labels, prob, name):
    """
    Creates a subnetwork that computes classification accuracy (fraction of correct top guesses in batch).

    :param labels: Ground truth labels.
    :param prob: Per-class probabilities output by the network.
    :param name: Name of the returned tensor.

    :returns: Tensor containing one scalar equal to the fraction of samples in batch where top (maximum
        probability) guess is correct.
    """

    # Hint:
    # Use elementary operations like
    # [`tf.argmax`](https://www.tensorflow.org/api_docs/python/tf/math/argmax),
    # [`tf.equal`](https://www.tensorflow.org/api_docs/python/tf/math/equal),
    # [`tf.reduce_mean`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean),
    # [`tf.cast`](https://www.tensorflow.org/api_docs/python/tf/dtypes/cast).

    return None

def get_conv_weights(graph, layer_name):
    """
    Returns weight tensor of the convolution operation within a given convolutional layer.

    :param model: Model graph containing the convolutional layer in question.
    :param layer_name: Name of the convolutional layer in question.

    :returns: Handle to 4D tensor containing weights of convolution operation.
    """

    # Hints:
    # - To fetch a tensor using its name use
    #   [tf.Graph.get_tensor_by_name](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).
    # - Consult TensorBoard graph visualization for name of the tensor in question. The name has the form
    #   <name of the node producing the tensor>:<output index>. For example, the name of the tensor
    #   containing the bias of conv1 is "conv1/bias:0".

    return None

def get_conv_outputs(graph, layer_name):
    """
    Returns output tensor of the convolution operation within a given convolutional layer.

    :param model: Model graph containing the convolutional layer in question.
    :param layer_name: Name of the convolutional layer in question.

    :returns: Handle to 4D tensor containing outputs of convolution operation (values before
        any bias or nonlinearity that may exist).
    """

    # Hints:
    # - To fetch a tensor using its name use
    #   [tf.Graph.get_tensor_by_name](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).
    # - Consult TensorBoard graph visualization for name of the tensor in question. The name has the form
    #   <name of the node producing the tensor>:<output index>. For example, the name of the tensor
    #   containing the bias of conv1 is "conv1/bias:0".

    return None

def image_summary(tensor, name):
    """
    Creates a subnetwork that computes a summary for a given batch of images.
    Uses [tf.summary.image](https://www.tensorflow.org/api_docs/python/tf/summary/image).

    :param tensor: Tensor of shape (<batch_size>, <height>, <width>) representing a batch of images.
    :param name: Name of the returned tensor.

    :returns: Tensor containing a summary object which represents the batch of images.
    """
    with tf.variable_scope(name):
        # Normalize to [0, 1].
        x_min = tf.reduce_min(tensor)
        x_max = tf.reduce_max(tensor)
        normalized = (tensor - x_min) / (x_max - x_min)
        return tf.summary.image("out", normalized, max_outputs=normalized.shape[0])

def conv_weight_summary(conv_weight_tensor, name):
    """
    Create a subnetwork that visualizes given tensor containing weights of a convolutional layer.

    :param conv_weight_tensor: Tensor to visualize.
    :param name: Name of the returned tensor.

    :returns: Tensor containing a summary object which represents a set of images depicting
        layer weights.
    """

    # Hints:
    # - Use provided `image_summary` function in this file.
    # - Remember to transpose the tensor appropriately. Weights are stored as tensors of shape
    #   (<kernel_height>, <kernel_width>, <num_input_channels>, <num_output_channels>).

    return None

def conv_output_summary(conv_output_tensor, example_index, name):
    """
    Create a subnetwork that visualizes given tensor containing outputs of a convolutional
    layer, restricted to a given example.

    :param conv_output_tensor: Tensor containing layer outputs for all examples.
    :param example_index: Index of the example whose outputs should be visualized.
    :param name: Name of the returned tensor.

    :returns: Tensor containing a summary object which represents a set of images depicting
        outputs of all filters on the example of interest.
    """

    # Hints:
    # - Use provided `image_summary` function in this file.
    # - Use [tf.slice](https://www.tensorflow.org/api_docs/python/tf/slice) to get required example.
    # - Remember to transpose the tensor appropriately. Convolutional layer outputs are laid out the
    #   same way as input images, see `image_placeholder`.

    return None
