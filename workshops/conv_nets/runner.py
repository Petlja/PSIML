import datetime
import numpy as np
import os
import tensorflow as tf

class Runner(object):
    """
    Evaluates given model on individual images, and visualizes the results.
    """

    def __init__(self, model):
        self.model = model
        with self.model.graph.as_default():
            # Create a session that will be used for all evaluations.
            self.session = tf.Session()

            # Specify device context. Here we use CPU. For information on using GPU, see
            # [this TensorFlow guide](https://www.tensorflow.org/tutorials/using_gpu).
            # The default device is a GPU if available, and a CPU otherwise, so we will omit device specification
            # from now on.
            with tf.device("/cpu:0") as dev:

                # Initialize parameters according to rules defined for them in the graph.
                self.session.run(tf.global_variables_initializer())

    def get_fetches(self):
        """
        Selects nodes to be evaluated.

        :returns: Dictionary from descriptive node IDs to nodes to be evaluated.
            See [`tf.Session.run`](https://www.tensorflow.org/api_docs/python/tf/Session#run).
        """
        # Advantage of using a dictionary over a list is being able to later reference results in a more
        # robust way (by ID instead of by index in the list).
        return {
            "guess_class" : self.model.guess_class,
            "guess_prob" : self.model.guess_prob,
            }

    def get_feed_dict(self, image):
        """
        Maps input image to network inputs.

        :param image: Input image.

        :returns: Dictionary that maps input nodes to data.
            See [`tf.Session.run`](https://www.tensorflow.org/api_docs/python/tf/Session#run).
        """
        # Note that a singleton batch axis is added.
        return {self.model.images : np.expand_dims(image, axis=0)}

    def get_predictions(self, result):
        """
        Computes network predictions from evaluation result.

        :param result: Evaluation result.

        :returns: A tuple consisting of a list of class indices, and a list of corresponding class probabilities.
        """
        # Remove singleton batch axis.
        return np.squeeze(result["guess_class"], axis=0), np.squeeze(result["guess_prob"], axis=0)

    def create_tensorboard_log(self, log_dir):
        """
        Creates TensorBoard log file, and writes visualization of current network to it.

        :param log_dir: Path to directory where log should be created.
        """

        # Hints:
        # - Use [`tf.summary.FileWriter`](https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter).
        # - Graph can be written to log by passing it to [`tf.summary.FileWriter`] constructor, or by
        #   invoking a dedicated method.

        pass

    def log_to_tensorboard(self, result):
        """
        Writes relevant part of evaluation result to TensorBoard log.

        :param result: Evaluation result.
        """

        # Hint:
        # Use [`tf.summary.FileWriter.add_summary`](https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter#add_summary).

        pass

    def run(self, image):
        """
        Computes current model's predictions on a given image using
        [`tf.Session.run`](https://www.tensorflow.org/api_docs/python/tf/Session#run).
        Current model is assumed to have guess_class and guess_prob properties returning nodes which compute
        class indices of top few predictions, and corresponding class probabilities.

        :param image: The input image.

        :returns: A tuple consisting of a list of class indices, and a list of corresponding class probabilities.
        """
        print("-" * 70)
        print("Running model.")
        print("-" * 70)

        # Evaluate network.
        with self.model.graph.as_default():
            result = self.session.run(fetches=self.get_fetches(), feed_dict=self.get_feed_dict(image))

        # Create new TensorBoard log for each invocation of this function.
        datetime_str = datetime.datetime.now().strftime('%Y-%m-%d %Hh%Mm%Ss')
        self.create_tensorboard_log(os.path.join(".", "logs", "runner", datetime_str))

        # Compute summaries, and write them to TensorBoard log.
        self.log_to_tensorboard(result)

        # Compute and return predictions.
        return self.get_predictions(result)

    def load(self, file_path):
        """
        Loads model parameters from checkpoint file on disk.

        :param file_path: Path to checkpoint file to be loaded.
        """

        # Hints:
        # - Use [`tf.train.Saver.restore`](https://www.tensorflow.org/api_docs/python/tf/train/Saver#restore).
        # - Implementation is analogous to `Trainer.save` in `trainer.py`.

        with self.model.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.session, file_path)
