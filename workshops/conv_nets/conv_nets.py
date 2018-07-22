# <markdowncell>

# # Convolutional Neural Network Workshop
#
# The aim of this workshop is to demonstrate the basics of using [TensorFlow](https://www.tensorflow.org/) for
# convolutional neural networks.
#
# TensorFlow is an open source system developed by Google Brain for working with computation graphs. It supports
# automatic differentiation, making it suitable for implementing backpropagation.
#
# Optimized implementation of convolutional network is not trivial, especially on GPU. Therefore, in your projects
# you are much more likely to use frameworks like TensorFlow than to code from scratch.
#
# This notebook will cover the basics, providing references to additional resources where needed. You are welcome
# to use many excellent tutorials on the web, including the [one from Google](https://www.tensorflow.org/get_started/get_started).

# Overview:
# * [Preparing data](#Preparing-data)
# * [Defining and examining a simple network](#Defining-and-examining-a-simple-network)
# * [Adding batch normalization](#Adding-batch-normalization)
# * [Training a more complex network](#Training-a-more-complex-network)
# * [Saving and loading models](#Saving-and-loading-models)
# * [Transfer learning](#Transfer-learning)

# <markdowncell>

# **Recommended TensorFlow version is 1.9.0.** If you are working on a machine with GPU, it is recommended that you
# **install GPU version of TensorFlow** (`tensorflow-gpu` Python package). You will also need `matplotlib`.
#
# We begin by importing all modules that we will need, including TensorFlow. This also verifies TensorFlow installation.

# <codecell>

import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from utils import load_cifar10, visualize_grid

# <markdowncell>

# ## Preparing data
#
# We will use CIFAR-10 dataset. Please download it from [here](https://www.cs.toronto.edu/~kriz/cifar.html)
# (alternatively from [here](https://1drv.ms/u/s!ApgLrbSPQF9XirQ0fEqljE5_IT0q6g)
# or [here](https://1drv.ms/u/s!ApgLrbSPQF9XirQ16VxWmz_x0bRwMg)), and make sure to specify correct location
# in `cifar10_dir` below. We load the dataset by calling a utility function.

# <codecell>

cifar10_dir = 'cifar-10-batches-py'
x_dev, y_dev, x_test, y_test = load_cifar10(cifar10_dir)

# <markdowncell>

# The dataset is divided into part that can be used for training and validation (which we will call
# *development*), and test part.
#
# There are 50,000 dev images, and 10,000 test images. test images. All images are 32 x 32 pixels, and
# have 3 channels. Each image is assigned to one of 10 classes.

# <codecell>

print('Dev data shape: ', x_dev.shape)
print('Dev labels shape: ', y_dev.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)
print('Class labels: ', np.unique(np.concatenate([y_dev, y_test])))

# <markdowncell>

# We define some useful constants to be used throughout

# <codecell>

num_dev = x_dev.shape[0]
num_test = x_test.shape[0]
num_classes = len(np.unique(np.concatenate([y_dev, y_test])))

# <markdowncell>

# ### Training/validation split
#
# From dev data we take 10% for validation, and use the rest for training.
# To make sure that train and dev set have the same distribution, we randomize
# dev set first.

# <codecell>

num_val = num_dev // 10
num_train = num_dev - num_val

perm = np.random.permutation(num_dev)
x_dev = x_dev[perm]
y_dev = y_dev[perm]

x_train = x_dev[:num_train]
y_train = y_dev[:num_train]
x_val = x_dev[num_train:]
y_val = y_dev[num_train:]

# <markdowncell>

# ### Data normalization
#
# We normalize all data by subtracting mean image from each sample.
# Note that the mean image is computed from training set only.

# <codecell>

mean_image = np.mean(x_train, axis=0)
x_train -= mean_image
x_val -= mean_image
x_test -= mean_image

# <markdowncell>

# ### Visualizing training examples

# Let's look at `sample_size` random examples from each class.

# <codecell>

sample_size = 10
sample = x_train[[i for c in range(num_classes)
                  for i in np.random.permutation(np.flatnonzero(y_train == c))[:sample_size]]]
sample += mean_image
plt.figure(figsize = (12, 12))
plt.imshow(visualize_grid(sample))
plt.axis("off")
plt.show()

# <markdowncell>

# Now that we know what the classes represent, we can create an array of human-readable names.

# <codecell>

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# <markdowncell>

# ## Defining and examining a simple network
#
# We define a simple network consisting of basic layers:
# * convolutional layer,
# * max-pooling layer,
# * fully connected layer, and
# * ReLU activation function.
#
# TensorFlow supports many other layer types and activations. See https://www.tensorflow.org/api_guides/python/nn for official API documentation. 

# <markdowncell>

# The following line clears any network that might already exist in memory. 

# <codecell>

tf.reset_default_graph()

# <markdowncell>

# ### Create TensorBoard log file
#
# We will use TensorBoard to visualize various data about our network. TensorBoard parses log files (also called event files) generated by TensorFlow. We will be placing those files in a separate dir.

# <codecell>

log_dir = './logs/'

# <markdowncell>

# A new event file is created by instantiating a `tf.FileWriter` class.

# <codecell>

writer = tf.summary.FileWriter(os.path.join(log_dir, 'simple_net'))

# <markdowncell>

# ### Placeholders for data
# First we define placeholders for input data (input image and its label) using `tf.placeholder`.
# We will eventually bind these to actual numerical data values.
#
# We choose to represent input data as 4D tensors whose shape is N x H x W x C, where:
# * N is the number of examples in a batch (batch size)
# * H is the height of each image in pixels
# * W is the height of each image in pixels
# * C is the number of channels (usually 3: R, G, B)
#
# This is the right way to represent the data for spatial operations like convolution. For fully connected layers, however, all dimensions except batch size will be collapsed into one.
#
# In `tf.placeholder`, if a dimension has value `None`, it will be set automatically once actual data is provided.

# <codecell>

def setup_input():
    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')
    y = tf.placeholder(tf.int64, [None], name='y')
    is_training = tf.placeholder(tf.bool, name='is_training')
    return x, y, is_training

# <codecell>

x, y, is_training = setup_input()

# <markdowncell>

# ### Convolutional and pooling nodes
# Next we start defining the main "body" of the network.
# We start by adding a single convolutional layer with bias and ReLU activation.
#
# We use [tf.layers API](https://www.tensorflow.org/api_docs/python/tf/layers) to generate a whole layer by a single function call.
# It is also possible to create each parameter and operation node separately, and connect them together, but that quickly becomes
# cumbersome for bigger networks.
#
# TensorFlow also provides other high-level APIs, such as [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim), which
# has become very popular, with a large number of [pretrained models](https://github.com/tensorflow/models/tree/master/research/slim) written in it.
# Fortunately, TF-Slim models can be quite easily combined with other TensorFlow APIs such as tf.layers.
#
# Convolutional layer is created by calling `tf.layers.conv2d`. Returned object is of type `tf.Tensor` and represents output activations of the layer.
#
# Bias is enabled by default, so it is not explicitly specified. `padding='SAME'` means that we allow padding of roughly half the kernel size
# (TensorFlow computes this value automatically), to avoid reduction in output size due to boundary effects. The other option is `padding='VALID'`, which means
# that padding is zero.

# <codecell>

conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[7, 7], strides=2, padding='SAME', activation=tf.nn.relu, name='conv1')

# <markdowncell>

# Next we add a max-pooling node.

# <codecell>

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='SAME', name='pool1')

# <markdowncell>

# ### View default graph in TensorBoard
#
# We can write graph data to event file we create above. A graph can be passed to `FileWriter` constructor as well, in which case it is written to file immediately after the file is created.

# <codecell>

writer.add_graph(tf.get_default_graph())

# <markdowncell>

# Now you should be able to run `tensorboard --logdir=./logs` from console (with your Python environment activated), and see the graph visualized in browser at `http://localhost:6006`.
#
# For more details please see official tutorial on [graph visualization](https://www.tensorflow.org/get_started/graph_viz).
#
# Note: graph visualization seems to work best in Google Chrome.

# <markdowncell>

# ### Examining static information
# Most information about the network which is static (i.e. independent of input data)
# can be seen from TensorBoard graph visualization -- node names, connectivity, tensor sizes,
# some layer parameters, but not tensor values.
#
# This data is also acessible from Python code, which is useful in some situations.
#
# If we already have handles on nodes, for example because we just created them, such as `conv1`
# and `pool1` above, we can query their properties.

# <codecell>

print(conv1.shape)
print(pool1.shape)

# <markdowncell>

# Note that shapes of activation tensors are computed automatically. Also, these tensors
# may have unknown dimensions which become known only when acutal input is presented.

# We may not have a handle on some nodes, for example internal parameters of `conv1` (kernel and bias).
# Those tensors are "hidden" inside the layer, because we used layers API. We can still access them
# using name lookup.

# <codecell>

conv1_kernel = tf.get_default_graph().get_tensor_by_name('conv1/kernel:0')
conv1_bias = tf.get_default_graph().get_tensor_by_name('conv1/bias:0')

# <markdowncell>

# The names can be found from TensorBoard visualization, or by listing all operations (nodes) in the network.

# <codecell>
tf.get_default_graph().get_operations()

# <markdowncell>

# The `conv1` prefix in both names (*name scope*) refers to the `name` parameter specified when creating
# the layer, and NOT the Python variable `conv1` that we assigned the result to. The `:0` suffix means that
# the tensor is the first (index 0) output of the node that produces it (`conv/kernel` and `conv1/bias`, respectively).
#
# We can get the shapes of kernel and bias as follows

# <codecell>

print(conv1_kernel.shape)
print(conv1_bias.shape)

# <markdowncell>

# Finally, we can also access internal parameters of operations (rather than their tensor outputs) using
# `get_operation_by_name` followed by `get_attr`.

# <codecell>

print(tf.get_default_graph().get_operation_by_name('conv1/Conv2D').get_attr('strides'))

# <markdowncell>

# ### Fully connected layers
# Next we append a fully connected layer with 1024 output neurons and ReLU activation.
# In order to determine the shape of its parameter tensor, we need to know the number of input neurons, which depends on the shape of the `relu1` activation tensor.

# <codecell>

fc1_input_count = int(pool1.shape[1] * pool1.shape[2] * pool1.shape[3])
fc1_output_count = 1024
print([fc1_input_count, fc1_output_count])

# <markdowncell>

# In order to append a fully connected layer, we need to flatten the spatial dimensions of `relu1`.

# <codecell>

pool1_flat = tf.reshape(pool1, [-1, fc1_input_count])

# <markdowncell>

# Now we are ready to add a fully connected layer.

# <codecell>

fc1 = tf.layers.dense(inputs=pool1_flat, units=fc1_output_count, activation=tf.nn.relu, name='fc1')

# <markdowncell>

# Finally, we add another fully connected layer with bias to output scores for 10 output classes. This layer has no nonlinearity following it, but it will be followed by a softmax function to convert scores to probabilities.

# <codecell>

fc2 = tf.layers.dense(inputs=fc1, units=num_classes, name='fc2')

# <markdowncell>

# ### Final classification
# We append a softmax layer to convert the scores coming from `fc2` into probabilities, as well as a "top-k" layer to get the three most probable guesses.

# <codecell>

prob = tf.nn.softmax(fc2)
(guess_prob, guess_class) = tf.nn.top_k(prob, k=3)

# <markdowncell>

# ### Visualizing parameters and activations
# TensorBoard supports visualizing tensors as images using `tf.summary.image` function.
# This function adds a subnetwork that computes images for a given tensor.

# <codecell>

def setup_image_summary(tensor, name):
    with tf.variable_scope(name):
        # Normalize to [0 1].
        x_min = tf.reduce_min(tensor)
        x_max = tf.reduce_max(tensor)
        normalized = (tensor - x_min) / (x_max - x_min)

        # Display random 3 slices.
        return tf.summary.image('tensor', normalized, max_outputs=3)

# <markdowncell>

# Using this we visualize some of the weights in `conv1_kernel`. Since each filter has 3
# input channels, it be visualized as an RGB image. Note that we need to transpose the
# tensor from (H, W, C, N) to (N, H, W, C) layout.

# <codecell>

conv1_kernel_summary = setup_image_summary(
    tf.transpose(conv1_kernel, [3, 0, 1, 2]),
    name='conv1_kernel_summary')

# <markdowncell>

# Similarly, we can visualize some of the activations in `conv1`. Use `setup_image_summary`
# in code below to visualize a subset of channels for example 0 in minibatch. Each channel
# should be visualized as a grayscale image. You can use [tf.slice](https://www.tensorflow.org/api_docs/python/tf/slice)
# to get example 0. If necessary, transpose into appropriate layout.

# <codecell>

# TODO

# <markdowncell>

# Summaries can also be *merged*. We add a special `image_summaries` node that outputs the union of our two image summaries.
# Evaluating this node causes both summaries to be computed.

# <codecell>

image_summaries = tf.summary.merge([conv1_kernel_summary, conv1_output_summary])

# <markdowncell>

# ### Update graph visualization
# We have added some new nodes, and we need to check if the new graph is OK.
# To update TensorBoard visualization, we just add a new graph to the event file.
# The visualizer will pick up the latest graph when its browser tab is refreshed.

# <codecell>

writer.add_graph(tf.get_default_graph())

# <markdowncell>

# ### Forward pass
# Next we run one CIFAR-10 frame through the network.

# <codecell>

def choose_random_image():
    index = np.random.randint(0, num_train)
    return index, x_train[[index]], y_train[[index]]

# <codecell>

random_index, random_image, random_label = choose_random_image()

# <markdowncell>

# A TensorFlow graph is executed by creating a `tf.Session` object and calling its `run` method.
# A session object encapsulates the control and state of the TensorFlow runtime.
# The `run` method requires a list of output tensors that should be computed, and a mapping of input tensors to actual data that should be used. For more information, see the TensorFlow [Getting started](https://www.tensorflow.org/get_started/get_started) guide.
#
# Optionally we can also specify a device context such as `/cpu:0` or `/gpu:0`. For documentation on this see [this TensorFlow guide](https://www.tensorflow.org/tutorials/using_gpu). The default device is a GPU if available, and a CPU otherwise, so we can skip the device specification from now on.
#
# Note: if GPU is explicitly specified, but not available, a Python exception is thrown; current graph is invalidated, and needs to be cleared and rebuilt.

# <codecell>

with tf.Session() as sess:
    with tf.device("/cpu:0") as dev: #"/cpu:0" or "/gpu:0"
        # Initialize weights.
        sess.run(tf.global_variables_initializer())

        # Map inputs to data.
        feed_dict = { x : random_image, y : random_label }

        # Set up variables we want to compute.
        variables = [guess_prob, guess_class, image_summaries]

        # Perform forward pass.
        guess_prob_value, guess_class_value, img_summ_value = sess.run(variables, feed_dict=feed_dict)

# <markdowncell>

# First let's see the image that was chosen, and networks predictions for it.

# <codecell>

def visualize_classification(image, guess_class, guess_prob):
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    for i in range(3):
        ind = guess_class[0, i]
        prob = guess_prob[0, i]
        print("Class: {0}\tProbability: {1:0.0f}%".format(class_names[ind], prob * 100))
    print("Ground truth: {0}".format(class_names[random_label[0]]))

# <codecell>

visualize_classification((random_image[0] + mean_image) / 255.0, guess_class_value, guess_prob_value)

# <markdowncell>

# We write generated images to file. After running the next cell the images should be visible in TensorBoard.

# <codecell>

writer.add_summary(img_summ_value)

# <markdowncell>

# ### Loss and metric(s)
#
# We append more nodes to compute loss value, and the number of correctly predicted pixels.
# For loss we use `tf.sparse_softmax_cross_entropy_with_logits`. For other loss functions available
# out of the box in TensorFlow, see https://www.tensorflow.org/api_guides/python/nn#Losses and
# https://www.tensorflow.org/api_guides/python/nn#Classification.
# Of course, you can always build your own custom loss functions from simpler operations.

# <codecell>

def setup_metrics(y, y_out):
    # Define loss function.
    total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_out)
    mean_loss = tf.reduce_mean(total_loss)

    # Add top three predictions.
    prob = tf.nn.softmax(y_out)
    (guess_prob, guess_class) = tf.nn.top_k(prob, k=3)

    # Compute number of correct predictions.
    is_correct = tf.equal(tf.argmax(y_out, 1), y)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    return mean_loss, accuracy, guess_prob, guess_class

# <markdowncell>

# We will be reusing this function later for other architectures.
# Now we create metrics for our current network.

# <codecell>

mean_loss, accuracy, guess_prob, guess_class = setup_metrics(y, fc2)

# <markdowncell>

# ### Visualizing loss and metric(s)
# We would like to use TensorBoard to visualize loss value and correct count.
# We add special nodes that generate those logs.

# <codecell>

def setup_scalar_summaries():
    mean_loss_summary = tf.summary.scalar('mean_loss', mean_loss)
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    return tf.summary.merge([mean_loss_summary, accuracy_summary])

# <codecell>

scalar_summaries = setup_scalar_summaries()

# <markdowncell>

# ### Optimizer
#
# Finally, we define the optimization algorithm to be used for training. We use the Adam optimizer with learning rate 5e-4. For other choices see https://www.tensorflow.org/api_guides/python/train#Optimizers.
#
# Optimizer's `minimize` method essentially generates a network that performs backward pass based on the forward pass network that we defined, and passed to the optimizer via argument to `minimize`.
# The result of this method is a dummy node `train_step` which, when evaluated triggers execution of backward pass.

# <codecell>

def setup_optimizer(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Batch normalization in TensorFlow requires this extra dependency
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(loss)
    return train_step

# <markdowncell>

# We will be reusing this function for other architectures. Now we create optimizer for our current network.

# <codecell>

train_step = setup_optimizer(mean_loss, 5e-4)

# <markdowncell>

# ### Adding an optional backward pass
# Above we saw how to execute forward pass using `tf.Session.run`. Now we wrap that into a function
# (since we will be calling it in a loop to train the network). We also add an option to execute a
# backward pass by passing the extra argument `training`. That way we can use the same function for
# both training (forward + backward), and evaluation (forward only).

# <codecell>

def run_iteration(session, variables, x_data, y_data, training=None):
    if training != None:
        variables += [training]

    # Map inputs to data.
    feed_dict = {x : x_data, y : y_data, is_training : training != None}

    # Compute variable values, and perform training step if required.
    values = session.run(variables, feed_dict=feed_dict)

    # Return loss value and number of correct predictions.
    return values[:-1] if training != None else values

# <markdowncell>

# ### Main training/evaluation loop
# The following is a simple function which trains or evaluates current model for a given number
# of epochs by repeatedly calling the `run_iteration` function defined above. It also takes care of:
# * aggregating loss and accuracy values over all minibatches
# * plotting loss and accuracy values over time.
#
# The code below assumes that `mean_loss`, `accuracy`, and `scalar_summaries` are defined externally.
# Optionally, the caller may also define `image_summaries`, if they want some image summaries (like kernel
# and activation visualizations in the form of images) to be also included.
# All those definitions will change depending on the network.

# <codecell>

def run_model(session, x, y, epochs, batch_size, print_every, training):

    # Number of examples in dataset.
    dataset_size = x.shape[0]

    # Count iterations since the beginning of training.
    iter_cnt = 0

    for e in range(epochs):
        # Randomize training examples for each epoch.
        train_indices = np.random.permutation(dataset_size)

        # Keep track of performance stats (loss and accuracy) in current epoch.
        total_loss = 0
        total_correct = 0

        # Iterate over the dataset once.
        for start_idx in range(0, dataset_size, batch_size):

            # Indices for current batch.
            idx = train_indices[start_idx : min(start_idx + batch_size, dataset_size)]

            # Get batch size (may not be equal to batch_size near the end of dataset).
            actual_batch_size = y[idx].shape[0]

            # Set up variables that we want to compute.
            variables = [mean_loss, accuracy, scalar_summaries]
            has_image_summaries = image_summaries != None
            if has_image_summaries:
                variables += [image_summaries,]

            # Compute loss, accuracy, and scalar summaries, and optionally perform backward pass.
            results = run_iteration(session, variables, x[idx], y[idx], training)

            # Unpack results.
            loss, acc, scl_summ = results[:3]
            if has_image_summaries:
                img_summ = results[3]

            # Update performance stats.
            total_loss += loss * actual_batch_size
            total_correct += acc * actual_batch_size

            # Add scalar summaries to event file.
            if (training is not None):
                writer.add_summary(scl_summ, iter_cnt)

            # Print status, and add image summaries (if any).
            if (training is not None) and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2f}%".format(iter_cnt, loss, acc * 100))
                if has_image_summaries:
                    writer.add_summary(img_summ, iter_cnt)

            iter_cnt += 1

        # Compute performance stats for current epoch.
        avg_accuracy = total_correct / dataset_size
        avg_loss = total_loss / dataset_size

        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.2f}%".format(avg_loss, avg_accuracy * 100, e + 1))

# <markdowncell>

# ### Training the model for one epoch

# <codecell>

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess, x=x_train, y=y_train, epochs=1, batch_size=64, print_every=100, training=train_step)
print('Validation')
run_model(sess, x=x_val, y=y_val, epochs=1, batch_size=64, print_every=100, training=None)

# <markdowncell>

# ### View summaries in TensorBoard log
# Now you should be able to refresh your TensorBoard tab and see the summaries.
# For more details please see [official tutorial](https://www.tensorflow.org/get_started/summaries_and_tensorboard) on summaries.
# TensorFlow also supports other kinds of summaries, such as [histograms](https://www.tensorflow.org/get_started/tensorboard_histograms). 

# <markdowncell>

# ### Visualize some predictions
# Accuracy should be somewhat better now.

# <codecell>

random_index, random_image, random_label = choose_random_image()
guess_class_value, guess_prob_value = run_iteration(sess, [guess_class, guess_prob], random_image, random_label)
visualize_classification((random_image[0] + mean_image) / 255.0, guess_class_value, guess_prob_value)

# <markdowncell>

# ## Adding batch normalization
#
# Now we will modify the simple architecture by adding batch normalization. We expect this network to train faster, and achieve better
# accuracy for the same number of weight updates.
#
# Your task is to implement `bn_net` function below so that it creates the same network as in the previous section (with convolution, pooling,
# and two fully connected layers), except that there should also be a batch normalization layer after the convolution layer.
# 
# Arguments of `bn_net` are placeholders for data (`x`), labels (`y`), and boolean flag (`is_training`) for indicating if the network should
# perform training or validation version of the computation on the input data. While `is_training` was completely ignored in the previous
# network, *here it should actually matter* (why?). `bn_net` should return the output (prediction) tensor of the network.
#
# API reference for batch normalization is at https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization.

# <codecell>

def bn_net(x, y, is_training):
    # TODO
    pass

# <markdowncell>

# Input, metrics and optimizer are the same as before, so we can assemble the whole network.
# If you implemented `bn_net` correctly, this code should run without modification.

# <codecell>

tf.reset_default_graph()
writer = tf.summary.FileWriter(os.path.join(log_dir, 'bn_net'))
x, y, is_training = setup_input()
y_out = bn_net(x, y, is_training)
mean_loss, accuracy, guess_prob, guess_class = setup_metrics(y, y_out)
scalar_summaries = setup_scalar_summaries()
image_summaries = None
train_step = setup_optimizer(mean_loss, 5e-4)

# <markdowncell>

# Now we are ready to train and validate the network with batch normalization as before.

# <codecell>

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Training')
    run_model(sess, x=x_train, y=y_train, epochs=1, batch_size=64, print_every=100, training=train_step)
    print('Validation')
    run_model(sess, x=x_val, y=y_val, epochs=1, batch_size=64, print_every=100, training=None)

# <markdowncell>

# ## Training a more complex network
#
# Please download TensorBoard event file from [here](https://1drv.ms/u/s!ApgLrbSPQF9XirQrPV7L1GOhxkc-ig).
# Your task is to build the architecture contained in the event file `cifar10_net_log\events.out.tfevents.1500743084.localhost`, and
# train it on CIFAR-10. You should train for 8 epochs with batch size 100 and learning rate 0.001.

# <codecell>

def cifar10_net(x, y, is_training):
    # TODO
    pass

# <codecell>

tf.reset_default_graph()
writer = tf.summary.FileWriter(os.path.join(log_dir, 'complex_net'))
x, y, is_training = setup_input()
y_out = cifar10_net(x, y, is_training)
mean_loss, accuracy, guess_prob, guess_class = setup_metrics(y, y_out)
scalar_summaries = setup_scalar_summaries()
image_summaries = None
train_step = setup_optimizer(mean_loss, 1e-3)

# <codecell>

writer.add_graph(tf.get_default_graph())

# <codecell>

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Training')
    run_model(sess, x=x_train, y=y_train, epochs=8, batch_size=100, print_every=100, training=train_step)
    print('Validation')
    run_model(sess, x=x_val, y=y_val, epochs=1, batch_size=100, print_every=100, training=None)

# <markdowncell>

# ## Saving and loading models
#
# Saving is done using `tf.train.Saver` class:
# * `save` method saves both network definition and weights.
# * `export_meta_graph` method saves only network definition.
#
# Loading is done in two stages:
# * `tf.train.import_meta_graph` function loads network definition, and returns a saver object that was used to save the model.
# * `restore` method of the returned saver object loads the weights.
#
# Note that since weights are available only inside a session, `save` and `restore` methods above require a session object as a parameter.
#
# Official TensorFlow documentation: [Saving and Restoring Variables](https://www.tensorflow.org/api_guides/python/state_ops#Saving_and_Restoring_Variables), [tf.train.Saver class](https://www.tensorflow.org/api_docs/python/tf/train/Saver), [tf.train.import_meta_graph function](https://www.tensorflow.org/api_docs/python/tf/train/import_meta_graph).
#
# Useful unofficial tutorial on saving and loading: http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
#
# Many pretrained models found online do not contain network definition (meta-graph) files. The authors instead provide TensorFlow
# code to create the network graph. For such graphs, weights can also be restored from checkpoint file. Network definition file is
# just a way to avoid copying the code.

# <markdowncell>

# ## Transfer learning
#
# In this section we will take a model which is pretrained on ImageNet for 1000-class image classification task, and
# finetune it for our 10-class CIFAR-10 classification task.
#
# Pretrained model is given by meta-graph file (containing network definition), and checkpoint file (containing weights).
# Please download pretrained model from [here](https://1drv.ms/u/s!ApgLrbSPQF9XirQpuN32y0577zey8g).

# <codecell>

pretrained_meta_graph = os.path.join('mobilenet_v2_1.0_224', 'mobilenet_v2_1.0_224.meta')
pretrained_checkpoint = os.path.join('mobilenet_v2_1.0_224', 'mobilenet_v2_1.0_224.ckpt')

# <markdowncell>

# For the CIFAR-10 task we need to perform the following two modifications to the pretrained model at the very minimum:
# * Process CIFAR-10 images so that their size becomes what pretrained model expects
# * Adapt the layers which perform final classification so that the number of output neurons is 10 (the number of
# classes in the CIFAR-10 classification task)

# <markdowncell>

# ### Get names of relevant nodes
#
# Modifying input part of a pretrained network is somewhat cumbersome. It must be done simultaneously with loading network
# definition, by passing to `tf.train.import_meta_graph` a mappping from input tensors of the pretrained network to new input tensors.
#
# First we load pretrained network definition only to get the names of input placeholder nodes that we want to replace.
# This step can be skipped if these names are already known.

# <codecell>

tf.reset_default_graph()
_ = tf.train.import_meta_graph(pretrained_meta_graph)

# <markdowncell>

# The easiest way to get the nodes' names is using TensorBoard. It can also be done programmatically, as explained above
# (for example, using `tf.get_default_graph().get_operations()`).

# <codecell>

writer = tf.summary.FileWriter(os.path.join(log_dir, 'transfer_net'))
writer.add_graph(tf.get_default_graph())

# <markdowncell>

# After inspecting TensorFlow graph visualization, we find that
# * the input nodes are `image` and `is_training`
# * final classification is performed in the node group (subgraph) `MobilenetV2/Logits/Conv2d_1c_1x1` which implements 1x1 convolution.

# <markdowncell>

# ### Modify input and output

# Next we clear the default graph, and start creating new one, with modified input subnetwork which upsamples input image
# to match the size expected by pretrained network.

# <codecell>

tf.reset_default_graph()
writer = tf.summary.FileWriter(os.path.join(log_dir, 'transfer_net'))
x, y, is_training = setup_input()
x_upsampled = tf.image.resize_images(x, [224, 224])

# <markdowncell>

# Finally, we reload pretrained network definition, replacing pretrained input placeholders with new tensors we just created.

# <codecell>

saver = tf.train.import_meta_graph(pretrained_meta_graph, input_map={'image:0' : x_upsampled, 'is_training:0' : is_training})

# <markdowncell>

# We want to replace the node group `MobilenetV2/Logits/Conv2d_1c_1x1` by a 1x1 convolution with different number of output
# channels. To that end, we get a handle to the tensor immediately preceding `MobilenetV2/Logits/Conv2d_1c_1x1` in the graph.

# <codecell>

feat = tf.get_default_graph().get_tensor_by_name('MobilenetV2/Logits/Dropout/cond/Merge:0')

# <markdowncell>

# Attach a new prediction layer for modified task.

# <codecell>

with tf.variable_scope('MobilenetV2/Logits'):
      conv_1x1 = tf.layers.conv2d(inputs=feat, filters=num_classes, kernel_size=[1, 1], strides=1, name='conv2d_1x1_modified')
      output = tf.squeeze(conv_1x1, [1, 2], name='output_modified')

# <markdowncell>

# ### Complete network definition
# Add metrics and optimizer as before.

# <codecell>

mean_loss, accuracy, guess_prob, guess_class = setup_metrics(y, output)
scalar_summaries = setup_scalar_summaries()
image_summaries = None
train_step = setup_optimizer(mean_loss, 5e-4)

# <markdowncell>

# Once again write out the graph to make sure surgery succeeded.

# <codecell>

writer.add_graph(tf.get_default_graph())

# <markdowncell>

# ### Train and validate network (at your own risk!)

# <markdowncell>

# Only now we can restore weights from checkpoint, because weights exist only inside a session.
#
# Then we train the network as before. Note that this training is *extremely slow on CPU*, due to bigger network.
# Also observe that we still need to initialize variables, because we have introduced new ones, i.e. not all
# variables that exist in the network are restored from checkpoint.

# <codecell>

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, pretrained_checkpoint)
    print('Training')
    run_model(sess, x_train, y_train, epochs=1, batch_size=64, print_every=100, training=train_step)
    print('Validation')
    run_model(sess, x_val, y_val, epochs=1, batch_size=64, print_every=100, training=None)
