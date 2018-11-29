import tensorflow as tf
import numpy as np
import multiprocessing
import cv2

# Names for features to be saved to tfrecord.
image_buffer_name = "image_buffer"
image_width_name = "image_width"
image_height_name = "image_height"
image_channels_name = "image_channels"
label_name = "label"

class TFRecordSampleParser(object):
    def __call__(self, example):
        # Create maping from feature name to its type description.
        feats = {
            image_buffer_name : tf.FixedLenFeature([], tf.string),
            image_width_name : tf.FixedLenFeature([], tf.int64),
            image_height_name : tf.FixedLenFeature([], tf.int64),
            image_channels_name : tf.FixedLenFeature([], tf.int64),
            label_name : tf.FixedLenFeature([], tf.int64)
        }

        # Take single example.
        features = tf.parse_single_example(example, feats)
        
        # Take features from example.
        image = tf.decode_raw(features["image_buffer"], tf.uint8)
        label = tf.cast(features["label"], tf.int32)
        height = tf.cast(features["image_height"], tf.int32)
        width = tf.cast(features["image_width"], tf.int32)
        channels = tf.cast(features["image_channels"], tf.int32)

        # Create actual image.
        image = tf.reshape(image, tf.stack([height, width, channels]))
        label = tf.reshape(label, (1,))

        # Return image and label.
        return image, label

class TFRecordReader(object):
    def __init__(self, tfrecord_path):
        # Count number of samples.
        self.count = 0
        for record in tf.io.tf_record_iterator(tfrecord_path):
            self.count += 1

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.parser = TFRecordSampleParser()

            cpu_count = multiprocessing.cpu_count()
            self.dataset = tf.data.TFRecordDataset(tfrecord_path)
            self.dataset = self.dataset.shuffle(buffer_size=10000)
            self.dataset = self.dataset.map(self.parser, num_parallel_calls=cpu_count)
            self.dataset = self.dataset.prefetch(buffer_size=2)
            self.dataset = self.dataset.repeat()
            self.iterator = self.dataset.make_initializable_iterator()
            self.next = self.iterator.get_next()

        with self.graph.as_default():
            self.session = tf.Session()

        self.session.run(self.iterator.initializer)

        image, label =  self.session.run(self.next)
        self.line_height = image.shape[0]

    def examples_count(self):
        return self.count
    
    def get_line_height(self):
        return self.line_height

    def get_data(self):
        image, label =  self.session.run(self.next)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.transpose()
        image = image / 255  # 1 - white, 0 - black
        image = image * (-1) + 1  # 0 - white, 1 - black
        return (image, label)