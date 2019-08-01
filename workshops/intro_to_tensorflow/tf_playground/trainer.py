import tensorflow as tf
import os
import numpy as np

import utils
from data import plot_data


class Trainer(object):
    def __init__(self, train_data, valid_data, model, epochs, batch_size, output):
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
        self.train_data = train_data
        self.valid_data = valid_data
        self.val_predictions = None
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
        self.val_predictions = None
        with self.model.graph.as_default():
            self.session.run(self.model.reset_accuracy)
            batch_count = self._batch_count(len(self.valid_data))
            for batch_id in range(batch_count):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size

                x = self.valid_data[batch_start:batch_end, 0]
                y = self.valid_data[batch_start:batch_end, 1]
                labels = self.valid_data[batch_start:batch_end, 2]

                _, accuracy, summary, predictions = self.session.run(
                    (self.model.update_accuracy, self.model.accuracy,
                     self.model.summary, self.model.outputs),
                    feed_dict={self.model.x: x, self.model.y: y,
                               self.model.labels: labels}
                )
                self.valid_summary_writer.add_summary(
                    summary, self._epochs_training)
                if self.val_predictions is None:
                    self.val_predictions = np.column_stack((x, y, predictions))
                else:
                    self.val_predictions = np.concatenate([
                        self.val_predictions, np.column_stack((x, y, predictions))])
                self.val_accuracy = accuracy
            self._save_val_predictions_plot()
            utils.progress_bar(self._epochs_training,
                               self.epochs, prefix='Epochs')

    def _save_val_predictions_plot(self):
        fig = plot_data(self.val_predictions)
        img = utils.plot_to_image(fig)
        summary = tf.Summary(value=[tf.Summary.Value(tag="Val predictions", 
            image=tf.Summary.Image(encoded_image_string=img, height=6, width=6))])
        self.valid_summary_writer.add_summary(summary, self._epochs_training)

    def _init_model(self):
        with self.model.graph.as_default():
            self.session.run([tf.global_variables_initializer(),
                              tf.tables_initializer()])

    def _batch_count(self, points_count):
        return points_count // self.batch_size

    def _train_epoch(self):
        np.random.shuffle(self.train_data)
        with self.model.graph.as_default():
            self.session.run(self.model.reset_accuracy)
            batch_count = self._batch_count(len(self.train_data))
            for batch_id in range(batch_count):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size

                x = self.train_data[batch_start:batch_end, 0]
                y = self.train_data[batch_start:batch_end, 1]
                labels = self.train_data[batch_start:batch_end, 2]

                _, _, summary = self.session.run(
                    (self.model.train_op, self.model.update_accuracy, self.model.summary),
                    feed_dict={self.model.x: x, self.model.y: y,
                               self.model.labels: labels}
                )
                self.train_summary_writer.add_summary(
                    summary, self._epochs_training)
            self._epochs_training += 1
