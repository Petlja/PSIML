import tensorflow as tf

import utils


class Model(object):
    def __init__(self):
        self.graph = tf.Graph()
        batch_size = None
        with self.graph.as_default():
            # Create input placeholders
            # TODO: look at the last workshop's placeholders,
            # you should add shape of placeholder for images
            self.images = tf.placeholder(
                tf.float32, shape=( ADD SHAPE ))
            self.labels = tf.placeholder(
                dtype=tf.int64, shape=(batch_size,), name='labels')

            # Create features
            with tf.name_scope('features'):
                # TODO: make sure that you have flat features
                # for the trainable layers
                # try searching a bit - tf is well documented :)
                # if stuck, ask for hints
                self.flat_image =

            # Add hidden layers
            with tf.name_scope('layers'):
                # TODO: you should add layers here
                # you can try one hidden layer used in tutorial:
                # https://www.tensorflow.org/tutorials/keras/basic_classification#setup_the_layers
                # just write it in tensorflow
                # or try and add your own ideas
                last_layer = 

            # Create predictions
            with tf.name_scope('output'):
                # TODO: bellow is the code from the previous workshop
                # you should make changes to support predicting 
                # 10 classes
                logits = tf.layers.dense(
                    inputs=last_layer, units=2, activation=None)
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
                # TODO: Add train operation
                # take a look at the previous workshop,
                # search for other optimizers
                # recommendation: adam
                self.train_op = 

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
