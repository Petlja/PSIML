import tensorflow as tf
import model_utils
import os
import numpy as np
from utils.progress import ProgressBar
import data.tokens

# Hyper-parameters
_WORD_EMBED_SIZE = 100
_RNN_UNITS = 32


class Model(object):
    def __init__(self, dictionary_size, embedding_size=_WORD_EMBED_SIZE, output_classes=3):
        self.graph = tf.Graph()
        batch_size = None
        with self.graph.as_default():
            # Create input placeholders
            self.word_ids = tf.placeholder(dtype=tf.int32, shape=(batch_size, None))
            self.word_ids_len = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
            self.labels = tf.placeholder(dtype=tf.int32, shape=(batch_size, 1))

            # Create word embeddings
            word_embeddings = tf.get_variable("word_embeddings", [dictionary_size, embedding_size])
            embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, self.word_ids)

            # Create RNN on top of word embeddings
            rnn_out = model_utils.rnn_vanilla(inputs=embedded_word_ids, units=_RNN_UNITS,
                                              sequence_length=self.word_ids_len)

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


def _reshape_word_ids_and_label(word_ids, word_ids_len, label, batch_size):
    word_ids = np.array(word_ids).reshape(batch_size, -1)
    word_ids_len = np.array(word_ids_len).reshape(batch_size)
    label = np.array(label).reshape(batch_size, -1)
    return word_ids, word_ids_len, label


def _get_padded_batch(tweet_data, from_elem, to_elem, padding_value):
    batch_word_ids = []
    batch_word_ids_len = []
    batch_labels = []
    for i in range(from_elem, to_elem):
        word_ids, label = tweet_data[i]
        word_ids, word_ids_len, label = _reshape_word_ids_and_label(word_ids, len(word_ids), label, batch_size=1)
        batch_word_ids.append(word_ids)
        batch_word_ids_len.append(word_ids_len)
        batch_labels.append(label)
    batch_word_ids_len = np.concatenate(batch_word_ids_len)
    max_len = max(batch_word_ids_len)
    batch_word_ids = [
        np.pad(word_ids, ((0, 0), (0, max_len - word_ids.size)), mode='constant', constant_values=padding_value)
        for word_ids in batch_word_ids]
    batch_word_ids = np.concatenate(batch_word_ids, axis=0)
    batch_labels = np.concatenate(batch_labels, axis=0)
    return batch_word_ids, batch_word_ids_len, batch_labels


class Trainer(object):
    def __init__(self, train_data, valid_data, word_dictionary, output):
        self.model = Model(dictionary_size=word_dictionary.dictionary_size())
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
        self.train_history = []
        self.valid_history = []
        self.early_stop_after = 1
        self.batch_size = 1024
        self._padding_id = word_dictionary.word_id(data.tokens.PADDING_WORD)

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
            batch_count = self._batch_count(len(self.valid_data))
            self._reset_accuracy()
            progress_bar = ProgressBar(total=batch_count, name="valid")
            for batch_id in range(batch_count):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_word_ids, batch_word_ids_len, batch_labels = _get_padded_batch(tweet_data=self.valid_data,
                                                                                     from_elem=batch_start,
                                                                                     to_elem=batch_end,
                                                                                     padding_value=self._padding_id)
                _, acc_val, loss, summary = self.session.run(
                    (self.model.acc_op, self.model.acc_value, self.model.loss, self.model.summary),
                    feed_dict={self.model.word_ids: batch_word_ids, self.model.labels: batch_labels,
                               self.model.word_ids_len: batch_word_ids_len}
                )
                self.valid_summary_writer.add_summary(summary)
                epoch_loss += loss * self.batch_size
                acc_str = "acc={0:.2f}".format(acc_val)
                progress_bar.show(batch_id, suffix=acc_str)
            tweet_count = batch_count * self.batch_size
            epoch_loss = epoch_loss / tweet_count
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

    def _batch_count(self, tweet_count):
        return tweet_count // self.batch_size

    def _train_epoch(self):
        self.train_data.shuffle()
        with self.model.graph.as_default():
            self._reset_accuracy()
            epoch_loss = 0
            batch_count = self._batch_count(len(self.train_data))
            progress_bar = ProgressBar(total=batch_count, name="train")
            for batch_id in range(batch_count):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_word_ids, batch_word_ids_len, batch_labels = _get_padded_batch(tweet_data=self.train_data,
                                                                                     from_elem=batch_start,
                                                                                     to_elem=batch_end,
                                                                                     padding_value=self._padding_id)
                _, _, acc, loss, summary = self.session.run(
                    (self.model.train_op, self.model.acc_op, self.model.acc_value, self.model.loss, self.model.summary),
                    feed_dict={self.model.word_ids: batch_word_ids, self.model.labels: batch_labels,
                               self.model.word_ids_len: batch_word_ids_len}
                )
                self.train_summary_writer.add_summary(summary)
                epoch_loss += loss * self.batch_size
                acc_str = "acc={0:.2f}".format(acc)
                progress_bar.show(batch_id, suffix=acc_str)
            tweet_count = batch_count * self.batch_size
            epoch_loss = epoch_loss / tweet_count
            loss_str = "loss={0:.2f}".format(epoch_loss)
            self.train_history.append(epoch_loss)
            progress_bar.total(suffix=acc_str + " " + loss_str)

    def _reset_accuracy(self):
        with self.model.graph.as_default():
            # Reset accuracy metrics
            self.session.run(self.model.acc_reset)


class Runner(object):
    def __init__(self, word_dictionary, checkpoint):
        self.model = Model(dictionary_size=word_dictionary.dictionary_size())
        with self.model.graph.as_default():
            self.session = tf.Session()
        model_utils.load_graph(self.model.graph, self.session, checkpoint)
        self._word_dictionary = word_dictionary

    def sentiment(self, tweet):
        word_ids = self._word_dictionary.word_ids(tweet)
        word_ids, word_ids_len, _ = _reshape_word_ids_and_label(word_ids, len(word_ids), label=1, batch_size=1)
        with self.model.graph.as_default():
            output = self.session.run(self.model.outputs,
                                      feed_dict={self.model.word_ids: word_ids, self.model.word_ids_len: word_ids_len})
            return output
