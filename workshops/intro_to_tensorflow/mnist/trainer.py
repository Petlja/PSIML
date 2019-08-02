import tensorflow as tf
import os
import numpy as np

import utils


class Trainer(object):
    def __init__(self, train_images, train_labels, valid_images, valid_labels, model, epochs, batch_size, output):
        tf.set_random_seed(5)
        self.model = model
        self.output = output
        self.model.save_graph_summary(os.path.join(self.output, 'summary'))
        self.train_summary_writer = \
            tf.summary.FileWriter(os.path.join(self.output, 'train'))
        self.valid_summary_writer = \
            tf.summary.FileWriter(os.path.join(self.output, 'valid'))
        with self.model.graph.as_default():
            self.session = tf.Session()
        self._init_model()
        self.train_images = train_images
        self.train_labels = train_labels
        self.valid_images = valid_images
        self.valid_labels = valid_labels
        self.val_accuracy = 0
        self._epochs_training = 0
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        while True:
            self._train_epoch()
            self.validate()
            if self._epochs_training == self.epochs:
                break

    def validate(self):
        with self.model.graph.as_default():
            self.session.run(self.model.reset_accuracy)
            batch_count = self._batch_count(len(self.valid_labels))
            for batch_id in range(batch_count):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size

                images = self.valid_images[batch_start:batch_end]
                labels = self.valid_labels[batch_start:batch_end]

                # TODO: take a look at the previous workshop
                # use session to run one validation epoch
                _, accuracy, summary, predictions = self.session.run(
                    (self.model.update_accuracy, self.model.accuracy,
                     self.model.summary, self.model.outputs),
                    feed_dict={self.model.images: images,
                               self.model.labels: labels}
                )  # WILL REMOVE

                self.valid_summary_writer.add_summary(
                    summary, self._epochs_training)
                self.val_accuracy = accuracy
            utils.progress_bar(self._epochs_training,
                               self.epochs, prefix='Epochs')

    def _init_model(self):
        with self.model.graph.as_default():
            self.session.run([tf.global_variables_initializer(),
                              tf.tables_initializer()])

    def _batch_count(self, points_count):
        return points_count // self.batch_size

    def _train_epoch(self):
        permutation = np.random.permutation(len(self.train_labels))
        self.train_images = self.train_images[permutation]
        self.train_labels = self.train_labels[permutation]
        with self.model.graph.as_default():
            self.session.run(self.model.reset_accuracy)
            batch_count = self._batch_count(len(self.train_labels))
            for batch_id in range(batch_count):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size

                images = self.train_images[batch_start:batch_end]
                labels = self.train_labels[batch_start:batch_end]

                # TODO: take a look at the previous workshop
                # use session to run one training epoch
                _, _, summary = self.session.run(
                    (self.model.train_op, self.model.update_accuracy, self.model.summary),
                    feed_dict={self.model.images: images,
                               self.model.labels: labels}
                )  # WILL REMOVE

                self.train_summary_writer.add_summary(
                    summary, self._epochs_training)
            self._epochs_training += 1

    def save_final_accuracy(self):
        self.valid_summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(
            tag="Final val accuracy", simple_value=self.val_accuracy)]))
