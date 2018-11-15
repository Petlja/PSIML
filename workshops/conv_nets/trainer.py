import datetime
import numpy as np
import os
import tensorflow as tf

class Trainer(object):
    """
    Trains and evaluates given model on batches of images, and visualizes training loss and accuracy.
    """

    def __init__(self, model, learning_rate):
        self.model = model

        with self.model.graph.as_default():
            # Create subnetwork that performs backward pass.
            # IMPORTANT: If current model contains batch normalization operations, their update operations need
            # to be explicitly added as dependencies of the returned update operation. This is because batch
            # normalization update operations are typically not in the subnetwork that influences the loss.
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = self.adam_optimizer(learning_rate)

            self.session = tf.Session()
            with tf.device("/cpu:0") as dev:
                # Run initializers on all variables.
                self.session.run(tf.global_variables_initializer())

    def adam_optimizer(self, learning_rate):
        """
        Creates a subnetwork that updates weights of current model using Adam update rule with
        given leaning rate, in order to minimize loss function of current model.

        :param learning_rate: The learning rate with which to apply Adam updates.

        :returns: The root node of the weight update subnetwork, i.e., the node whose evaluation
            performs backward pass and weight update.
        """

        # Hint:
        # Use [`tf.train.AdamOptimizer`](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer#minimize).

        return tf.train.AdamOptimizer(learning_rate).minimize(self.model.loss)

    def get_fetches(self, is_training):
        """
        Selects nodes to be evaluated.

        :param is_training: Flag indicating if network is being trained or evaluated.

        :returns: Dictionary from descriptive node IDs to nodes to be evaluated.
            See [`tf.Session.run`](https://www.tensorflow.org/api_docs/python/tf/Session#run).
        """

        # Clarification:
        # Advantage of using a dictionary over a list is being able to later reference results in a more
        # robust way (by ID instead of by index in the list).

        # Hint:
        # Remember to evaluate the weight update node when training.

        fetches = {
            "loss" : self.model.loss,
            "accuracy" : self.model.accuracy,
            }
        if is_training:
            # If training, also evaluate the weight update node.
            fetches["train_op"] = self.train_op
        return fetches

    def get_feed_dict(self, images, labels, is_training):
        """
        Maps input data to network inputs.

        :param images: Batch of images.
        :param labels: Corresponding batch of labels.
        :param is_training: Flag indicating if network is being trained or evaluated.

        :returns: Dictionary that maps input nodes to data.
            See [`tf.Session.run`](https://www.tensorflow.org/api_docs/python/tf/Session#run).
        """
        return {self.model.images : images, self.model.labels : labels, self.model.is_training : is_training}

    def get_loss_and_accuracy(self, result):
        """
        Computes loss and accuracy values for current batch from evaluation result.

        :param result: Evaluation result.

        :returns: A tuple consisting of loss and accuracy values for the batch.
        """
        return result["loss"], result["accuracy"]

    def run_minibatch(self, images, labels, is_training):
        """
        Depending on a flag, runs either a forward pass only (evaluation) or forward pass, backward
        pass, and weight update (training) on a given batch.
        Current model is assumed to have `loss` and `accuracy` properties returning nodes which compute
        loss and accuracy values.

        :param images: Batch of images.
        :param labels: Corresponding batch of labels.
        :param is_training: Flag indicating if network is being trained or evaluated.

        :returns: A tuple consisting of loss and accuracy values for the batch.
        """
        with self.model.graph.as_default():
            result = self.session.run(
                fetches=self.get_fetches(is_training),
                feed_dict=self.get_feed_dict(images, labels, is_training))
        return self.get_loss_and_accuracy(result)

    def run_epoch(self, images, labels, batch_size, print_every, is_training):
        """
        Depending on a flag, runs either a forward pass only (evaluation) or both forward and
        backward pass (training) on a given dataset for one pass over the dataset (epoch).

        :param images: Dataset images.
        :param labels: Dataset labels.
        :param batch_size: Batch size.
        :param print_every: The number of training batches between two status updates on console and log.
        :param is_training: Flag indicating if network is being trained or evaluated.

        :returns: Loss and accuracy for the epoch, aggregated over all batches.
        """

        # Number of examples in dataset.
        dataset_size = images.shape[0]

        if is_training:
            # Randomize training examples for each epoch.
            example_order = np.random.permutation(dataset_size)
        else:
            example_order = np.arange(dataset_size)

        # Keep track of performance statistics in current epoch.
        epoch_loss = 0
        epoch_accuracy = 0

        # Iterate over the dataset once.
        for batch_index in range(int(dataset_size / batch_size)):

            # Indices for current batch.
            batch_begin = batch_index * batch_size
            batch_end = min(batch_begin + batch_size, dataset_size)
            batch_examples = example_order[batch_begin : batch_end]

            # Get batch size (may not be equal to batch_size near the end of dataset).
            actual_batch_size = batch_end - batch_begin

            batch_loss, batch_accuracy = self.run_minibatch(images[batch_examples], labels[batch_examples], is_training)

            # Print statistics for current batch.
            if batch_index % print_every == 0:
                print("Batch %d: loss = %.3f, accuracy = %.2f%%" % (batch_index, batch_loss, batch_accuracy * 100))

            # Update epoch statistics.
            epoch_loss += batch_loss * actual_batch_size
            epoch_accuracy += batch_accuracy * actual_batch_size

        # Compute performance stats for epoch.
        epoch_loss = epoch_loss / dataset_size
        epoch_accuracy = epoch_accuracy / dataset_size

        return epoch_loss, epoch_accuracy

    def create_tensorboard_log(self, train_log_dir_path, val_log_dir_path):
        """
        Creates two new TensorBoard log files (also called event files) in two given
        folders, corresponding to training and validation parts of current training session.
        Also stores current model graph in both logs files.

        :param train_log_dir_path: Path to directory to store logs from training set.
        :param val_log_dir_path: Path to directory to store logs from validation set.
        """

        # Hints:
        # - Use [`tf.summary.FileWriter`](https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter).
        # - Graph can be written to log by passing it to [`tf.summary.FileWriter`] constructor, or by
        #   invoking a dedicated method.

        self.writer_train = tf.summary.FileWriter(logdir=train_log_dir_path, graph=self.model.graph)
        self.writer_val = tf.summary.FileWriter(logdir=val_log_dir_path, graph=self.model.graph)

    def get_summary(self, loss, accuracy):
        """
        Computes summary for given loss and accuracy values.

        :param loss: The loss value.
        :param accuracy: The accuracy value.

        :returns: A summary containing given loss and accuracy values as two scalars.
        """

        # Hints:
        # - Use [`tf.Summary`](https://www.tensorflow.org/api_docs/python/tf/Summary)
        #   and [`tf.Summary.Value`](https://www.tensorflow.org/api_docs/python/tf/summary/Summary/Value).
        # - [`tf.Summary`] has a `value` field which is a list of instances of [`tf.Summary.Value`].
        # - [`tf.Summary.Value`] has a `simple_value` field which is used for storing scalars.

        return tf.Summary(value=[
            tf.Summary.Value(tag="loss", simple_value=loss),
            tf.Summary.Value(tag="accuracy", simple_value=accuracy),
            ])

    def log_to_tensorboard(self, epoch, summary_train, summary_val):
        """
        Adds new training and validation data to TensorBoard log.

        :param epoch: 0-based index of current training epoch.
        :param summary_train: The training summary to write.
        :param summary_val: The validation summary to write.
        """

        # Hint:
        # Use [`tf.summary.FileWriter.add_summary`](https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter#add_summary).

        self.writer_train.add_summary(summary_train, epoch)
        self.writer_val.add_summary(summary_val, epoch)

    def train(self, images_train, labels_train, images_val, labels_val, batch_size, epochs, print_every):
        """
        Trains network for a given number of passes over given training set (epochs). After each epoch
        validates current network on given validation set. Logs loss and accuracy values after each epoch.

        :param images_train: Training images.
        :param labels_train: Training labels.
        :param images_val: Validation images.
        :param labels_val: Validation labels.
        :param batch_size: Batch size.
        :param epochs: The number of epochs.
        :param print_every: The number of training batches between two status updates on console and log.
        """
        print("-" * 70)
        print("Training model.")
        print("-" * 70)

        # Create new TensorBoard log for each invocation of this function.
        datetime_str = datetime.datetime.now().strftime('%Y-%m-%d %Hh%Mm%Ss')
        self.create_tensorboard_log(
            train_log_dir_path=os.path.join(".", "logs", "trainer", datetime_str, "train"),
            val_log_dir_path=os.path.join(".", "logs", "trainer", datetime_str, "val")
            )

        for epoch in range(epochs):
            # Train network for one epoch.
            print("Starting training epoch %d" % (epoch + 1,))
            loss_train, accuracy_train = self.run_epoch(images_train, labels_train, batch_size, print_every, is_training=True)
            print("End of training epoch %d: loss = %.3f, accuracy = %.2f%%" % (epoch + 1, loss_train, accuracy_train * 100))

            # Run current network on validation set.
            print("Starting validation after epoch %d" % (epoch + 1,))
            loss_val, accuracy_val = self.run_epoch(images_val, labels_val, batch_size, print_every, is_training=False)
            print("End of validation after epoch %d: loss = %.3f, accuracy = %.2f%%" % (epoch + 1, loss_val, accuracy_val * 100))

            # Compute summaries, and write them to TensorBoard log.
            summary_train = self.get_summary(loss_train, accuracy_train)
            summary_val = self.get_summary(loss_val, accuracy_val)
            self.log_to_tensorboard(epoch, summary_train, summary_val)

    def evaluate(self, images, labels, batch_size, print_every):
        """
        Evaluates network on given dataset.

        :param images: Evaluation images.
        :param labels: Evaluation labels.
        :param batch_size: Batch size.
        :param print_every: The number of batches between two status updates on console.
        """
        print("-" * 70)
        print("Evaluating model.")
        print("-" * 70)

        print("Starting evaluation")
        loss, accuracy = self.run_epoch(images, labels, batch_size, print_every, is_training=False)
        print("End of evaluation: loss = %.3f, accuracy = %.2f%%" % (loss, accuracy * 100))

    def save(self, file_path):
        """
        Saves model parameters to checkpoint file on disk.

        :param file_path: Path to checkpoint file to be created.
        """

        # Hints:
        # - Use [`tf.train.Saver.save`](https://www.tensorflow.org/api_docs/python/tf/train/Saver#save).
        # - Since weights are available only inside a session, `tf.train.Saver.save` requires a session object
        #   as a parameter.

        with self.model.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.session, file_path)
