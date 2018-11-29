import tensorflow as tf
import model_utils
import os
import numpy as np
from utils.progress import ProgressBar
from utils.io import ensure_parent_exists
from PIL import Image
import tfrecord_reader

# Hyper-parameters
_RNN_UNITS = 100


class Model(object):
    def __init__(self, line_height, output_classes=4):
        self.graph = tf.Graph()
        self.output_classes = output_classes
        batch_size = None
        with self.graph.as_default():
            # Create input placeholders
            self.line_image = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, line_height))
            self.line_width = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
            self.labels = tf.placeholder(dtype=tf.int32, shape=(batch_size, 1))

            # Create RNN
            rnn_out = model_utils.rnn_vanilla(inputs=self.line_image, units=_RNN_UNITS,
                                              sequence_length=self.line_width)

            # Create predictions
            logits = tf.layers.dense(inputs=rnn_out, units=output_classes, activation=None)
            probabilities = tf.nn.softmax(logits=logits, axis=-1)
            self.outputs = tf.argmax(probabilities, axis=-1)

            # Create loss
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=logits)

            # Create accuracy node
            self.acc_value, self.acc_op, self.acc_reset = model_utils.accuracy(
                labels=self.labels, predictions=self.outputs)

            # Create summary for monitoring training progress
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("acc", self.acc_value)
            self.summary = tf.summary.merge_all()

            # Create training operation
            self.train_op = model_utils.get_train_op(self.loss, learning_rate=1e-4)

    def save_graph_summary(self, file):
        with self.graph.as_default():
            model_utils.save_graph_summary(file)


def _prepare_data(line, label):
    line = np.expand_dims(line, axis=0)  # 1 x W x H
    line_width = np.array([line.shape[1]])
    label = np.expand_dims(label, axis=0)  # 1 x 1
    return line, line_width, label


def _pad_line(line, width, padding_value=0):
    assert len(line.shape) == 3
    n, w, h = line.shape
    to_pad = width - w
    assert n == 1
    return np.pad(line, pad_width=((0, 0), (0, to_pad), (0, 0)), mode='constant', constant_values=padding_value)


def _get_padded_batch(image_data, from_elem, to_elem, padding_value):
    batch_line = []
    batch_line_width = []
    batch_labels = []
    for i in range(from_elem, to_elem):
        line, label = image_data.get_data()
        line, line_width, label = _prepare_data(line, label)
        batch_line.append(line)
        batch_line_width.append(line_width)
        batch_labels.append(label)
    batch_line_width = np.concatenate(batch_line_width)
    max_width = max(batch_line_width)
    batch_line = [_pad_line(line, max_width, padding_value) for line in batch_line]
    batch_line = np.concatenate(batch_line, axis=0)
    batch_labels = np.concatenate(batch_labels, axis=0)
    return batch_line, batch_line_width, batch_labels


class Trainer(object):
    def __init__(self, train_tfrecord_path, valid_tfrecord_path, output):
        self.train_data = tfrecord_reader.TFRecordReader(tfrecord_path=train_tfrecord_path)
        self.valid_data = tfrecord_reader.TFRecordReader(tfrecord_path=valid_tfrecord_path)
        self.model = Model(line_height=self.train_data.get_line_height())
        self.output = output
        self.model.save_graph_summary(os.path.join(self.output, 'summary'))
        self.train_summary_writer = \
            tf.summary.FileWriter(os.path.join(self.output, 'train'))
        self.valid_summary_writer = \
            tf.summary.FileWriter(os.path.join(self.output, 'valid'))
        with self.model.graph.as_default():
            self.session = tf.Session()
        self._init_model()
        self.train_history = []
        self.valid_history = []
        self.early_stop_after = 4
        self.batch_size = 32

    def train(self):
        while True:
            self._train_epoch()
            self.validate()
            assert len(self.valid_history) == len(self.train_history)
            epoch = len(self.train_history) - 1
            self.save(Trainer._checkpoint(self.output, epoch))
            if self._early_stop():
                break

    def validate(self):
        with self.model.graph.as_default():
            epoch_loss = 0
            batch_count = self._batch_count(self.valid_data.examples_count())
            self._reset_accuracy()
            progress_bar = ProgressBar(total=batch_count, name="valid")
            for batch_id in range(batch_count):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_line, batch_line_width, batch_labels = _get_padded_batch(image_data=self.valid_data,
                                                                               from_elem=batch_start,
                                                                               to_elem=batch_end,
                                                                               padding_value=0)
                _, acc_val, loss, summary = self.session.run(
                    (self.model.acc_op, self.model.acc_value, self.model.loss, self.model.summary),
                    feed_dict={self.model.line_image: batch_line, self.model.labels: batch_labels,
                               self.model.line_width: batch_line_width}
                )
                self.valid_summary_writer.add_summary(summary)
                epoch_loss += loss * self.batch_size
                acc_str = "acc={0:.2f}".format(acc_val)
                progress_bar.show(batch_id, suffix=acc_str)
            example_count = batch_count * self.batch_size
            epoch_loss = epoch_loss / example_count
            loss_str = "loss={0:.2f}".format(epoch_loss)
            self.valid_history.append(epoch_loss)
            progress_bar.total(suffix=acc_str + " " + loss_str)

    def save(self, checkpoint):
        model_utils.save_graph(self.model.graph, self.session, checkpoint)

    def load(self, checkpoint):
        model_utils.load_graph(self.model.graph, self.session, checkpoint)

    @staticmethod
    def _checkpoint(output, epoch):
        return os.path.join(output, "model", "{0}.ckpt".format(epoch))

    def _early_stop(self):
        index_min = np.argmin(self.valid_history)
        return index_min + self.early_stop_after < len(self.valid_history)

    def _init_model(self):
        with self.model.graph.as_default():
            self.session.run([tf.global_variables_initializer(),
                              tf.tables_initializer()])

    def _batch_count(self, data_length):
        return data_length // self.batch_size

    def _train_epoch(self):
        with self.model.graph.as_default():
            self._reset_accuracy()
            epoch_loss = 0
            batch_count = self._batch_count(self.train_data.examples_count())
            progress_bar = ProgressBar(total=batch_count, name="train")
            for batch_id in range(batch_count):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_line, batch_line_width, batch_labels = _get_padded_batch(image_data=self.train_data,
                                                                                     from_elem=batch_start,
                                                                                     to_elem=batch_end,
                                                                                     padding_value=0)
                _, _, acc, loss, summary = self.session.run(
                    (self.model.train_op, self.model.acc_op, self.model.acc_value, self.model.loss, self.model.summary),
                    feed_dict={self.model.line_image: batch_line, self.model.labels: batch_labels,
                               self.model.line_width: batch_line_width}
                )
                self.train_summary_writer.add_summary(summary)
                epoch_loss += loss * self.batch_size
                acc_str = "acc={0:.2f}".format(acc)
                progress_bar.show(batch_id, suffix=acc_str)
            example_count = batch_count * self.batch_size
            epoch_loss = epoch_loss / example_count
            loss_str = "loss={0:.2f}".format(epoch_loss)
            self.train_history.append(epoch_loss)
            progress_bar.total(suffix=acc_str + " " + loss_str)

    def _reset_accuracy(self):
        with self.model.graph.as_default():
            # Reset accuracy metrics
            self.session.run(self.model.acc_reset)


class Runner(object):
    def __init__(self, checkpoint, dataset_path, output):
        self.dataset_data = tfrecord_reader.TFRecordReader(tfrecord_path=dataset_path)
        self.output = output
        self.model = Model(line_height=self.dataset_data.get_line_height())
        with self.model.graph.as_default():
            self.session = tf.Session()
        model_utils.load_graph(self.model.graph, self.session, checkpoint)
        self.batch_size = 32

    def _batch_count(self, data_length):
        return data_length // self.batch_size

    def analyze(self):
        # Initialize confusion matrix.
        confusion = np.zeros((self.model.output_classes, self.model.output_classes))

        with self.model.graph.as_default():
            batch_count = self._batch_count(self.dataset_data.examples_count())
            progress_bar = ProgressBar(total=batch_count, name="analyze")
            for batch_id in range(batch_count):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_line, batch_line_width, batch_labels = _get_padded_batch(image_data=self.dataset_data,
                                                                               from_elem=batch_start,
                                                                               to_elem=batch_end,
                                                                               padding_value=0)
                outputs = self.session.run(
                    self.model.outputs,
                    feed_dict={self.model.line_image: batch_line, self.model.line_width: batch_line_width}
                )

                for sample_id in range(self.batch_size):
                    target = batch_labels[sample_id]
                    prediction = outputs[sample_id]

                    # Update confusion matrix.
                    confusion[target, prediction] += 1

                    # Dump incorrect images.
                    out_image_path = os.path.join(
                        self.output,
                        'target_%d_prediction_%d' % (target, prediction),
                        'batch_%d_sample_%d.png' % (batch_id, sample_id)
                        )
                    numpy_img = 255.0 * (1.0 - np.transpose(batch_line[sample_id]))
                    pil_img = Image.fromarray(numpy_img).convert('L')
                    ensure_parent_exists(out_image_path)
                    pil_img.save(out_image_path)

                progress_bar.show(batch_id)

            # Output confusion matrix.
            np.savetxt(
                os.path.join(self.output, 'confusion_counts.tsv'),
                confusion,
                fmt='%5d',
                delimiter='\t'
                )
            np.savetxt(
                os.path.join(self.output, 'confusion_fractions.tsv'),
                confusion / np.sum(confusion, axis=1, keepdims=True),
                fmt='%.3f',
                delimiter='\t')
