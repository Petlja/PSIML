import model_utils

import tensorflow as tf

class Model(object):

    def __init__(self):
        """
        Defines a simple network that showcases the following basic layers:
        - Convolutional layer.
        - Max-pooling layer.
        - Fully connected layer.
        - ReLU activation function.
        TensorFlow supports many other layers and activations, see
        [official API documentation](https://www.tensorflow.org/api_guides/python/nn).
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.add_classifier_net()
            self.add_loss_and_accuracy()
            self.add_summary()

    def add_classifier_net(self):
        """
        Creates a convolutional network that classifies input image.
        """
        # Placeholder for input batch of images. It will later be bound to actual image data.
        self.images = model_utils.image_placeholder(height=32, width=32, channels=3, name="images")

        # The main "body" of the network, consisting of:
        # - Convolutional layer.
        # - Pooling layer.
        # - Fully connected layer with ReLU activation.
        # - Fully connected layer without activation, which outputs a score for each of 10 classes.
        self.conv1 = model_utils.conv_layer(inputs=self.images, filters=32, kernel_size=[7, 7], strides=2, with_activation=True, name="conv1")
        self.pool1 = model_utils.pool_layer(inputs=self.conv1, pool_size=[2, 2], strides=2, name="pool1")
        self.fc1 = model_utils.fc_layer(inputs=self.pool1, units=1024, with_activation=True, name="fc1")
        self.fc2 = model_utils.fc_layer(inputs=self.fc1, units=10, with_activation=False, name="fc2")

        # Creates a subnetwork that computes class predictions and their probabilities.
        # Softmax layer to convert scores to probabilities.
        self.prob = tf.nn.softmax(self.fc2, name="prob")

        # "Top-k" layer to get three most probable guesses, and their probabilities.
        (self.guess_prob, self.guess_class) = tf.nn.top_k(self.prob, k=3, name="top_k")

    def add_loss_and_accuracy(self):
        """
        Creates a subnetwork that computes cross-entropy loss and the fraction of correctly predicted samples.
        """
        self.labels = model_utils.label_placeholder(name="labels")
        self.loss = model_utils.cross_entropy_loss(labels=self.labels, logits=self.fc2, name="loss")
        self.accuracy = model_utils.classification_accuracy(labels=self.labels, prob=self.prob, name="accuracy")

    def add_summary(self):
        """
        Creates a subnetwork that produces a summary with the following data:
        - 0-th input image from current batch.
        - Weights of each filter of the `conv1` layer.
        - Outputs of the same filters for the 0-th input image from current batch.
        """

        # Index of the filter of interest in the `conv1` layer.
        example_index = 0

        # Get input image of interest.
        image = tf.slice(self.images, begin=[example_index, 0, 0, 0], size=[1, -1, -1, -1])

        # Get a handle on the weight and output tensors of interest.
        weights = model_utils.get_conv_weights(self.graph, "conv1")
        outputs = model_utils.get_conv_outputs(self.graph, "conv1")

        summary_list = [
            model_utils.image_summary(image, name="image"),
            model_utils.conv_weight_summary(weights, name="conv1_weights"),
            model_utils.conv_output_summary(outputs, example_index, name="conv1_outputs")
            ]
        self.summary = tf.summary.merge([s for s in summary_list if s is not None])
