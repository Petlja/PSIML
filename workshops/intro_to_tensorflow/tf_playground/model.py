import tensorflow as tf

import utils


class Model(object):
    def __init__(self):
        self.graph = tf.Graph()
        batch_size = None
        with self.graph.as_default():
            # Create input placeholders
            self.x = tf.placeholder(
                dtype=tf.float32, shape=(batch_size,), name='x')
            self.y = tf.placeholder(
                dtype=tf.float32, shape=(batch_size,), name='y')
            self.labels = tf.placeholder(
                dtype=tf.int64, shape=(batch_size,), name='labels')

            # Create features
            with tf.variable_scope('features'):
                # Try adding more features x^2, sin(x)...
                features_to_stack = [self.x, self.y]
                self.features = tf.stack(features_to_stack, axis=1)
                self.features.set_shape([None, len(features_to_stack)])

            # Add hidden layers
            with tf.variable_scope('layers'):
                layer1 = tf.layers.dense(
                    inputs=self.features, units=3, activation=None)

            # Create predictions
            with tf.variable_scope('output'):
                logits = tf.layers.dense(
                    inputs=layer1, units=2, activation=None)
                probabilities = tf.nn.softmax(logits=logits, axis=-1)
                self.outputs = tf.argmax(probabilities, axis=-1)

            # Create loss
            with tf.name_scope('loss'):
                cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                    labels=self.labels, logits=logits)
                self.loss = tf.reduce_mean(
                    cross_entropy, name='cross_entropy_loss')

            # Create training operation
            with tf.name_scope('train'):
                self.train_op = tf.train.GradientDescentOptimizer(
                    learning_rate=0.03).minimize(self.loss)

            # Create accuracy node
            with tf.name_scope('accuracy'):
                # We want to track accuracy through one epoch
                self.accuracy, self.update_accuracy = tf.metrics.accuracy(
                    labels=self.labels, predictions=self.outputs)
                vars = tf.contrib.framework.get_variables(
                    'accuracy', collection=tf.GraphKeys.LOCAL_VARIABLES)
                # and reset it when we're done
                self.reset_accuracy = tf.variables_initializer(vars)

            # Create summary for monitoring training progress
            with tf.name_scope('summary'):
                tf.summary.scalar("loss", self.loss)
                tf.summary.scalar("acc", self.accuracy)
                self.summary = tf.summary.merge_all()

    def save_graph_summary(self, summary_file):
        with self.graph.as_default():
            utils.ensure_parent_exists(summary_file)
            summary_writer = tf.summary.FileWriter(summary_file)
            summary_writer.add_graph(tf.get_default_graph())
            summary_writer.flush()
