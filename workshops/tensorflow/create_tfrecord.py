import tensorflow as tf
import argparse
import os
import cv2

# Array of supported languages.
languages = ["ko", "el", "en", "sr"]

# Maps each language to network output.
language_to_output = {
    "ko" : 0,
    "el" : 1,
    "en" : 2,
    "sr" : 3
}

# Names for features to be saved to tfrecord.
image_buffer_name = "image_buffer"
image_width_name = "image_width"
image_height_name = "image_height"
image_channels_name = "image_channels"
label_name = "label"

def Check(cond, message):
    """
    Verifies that given condition is true. If not raises exception with given message.

    Args:
        cond :      Condition to be evaluated.
        message :   Message to be used if cond is not true.
    """
    if not cond:
        raise Exception(message)

def check_tfrecord(tfrecord_path):
    """
    Performs verification of the tf record at the given path. Verification resolves to loading one example
    and checking spatial dimensions matching.

    Args:
        tfrecord_path : Path to tfrecord to be checked.
    """
    with tf.Session() as sess:
        # Describe features as a mapping from feature name to feature length.
        feats = {
            image_buffer_name : tf.FixedLenFeature([], tf.string),
            image_width_name : tf.FixedLenFeature([], tf.int64),
            image_height_name : tf.FixedLenFeature([], tf.int64),
            image_channels_name : tf.FixedLenFeature([], tf.int64),
            label_name : tf.FixedLenFeature([], tf.int64)
        }

        # Open tfrecord for reading.
        filename_queue = tf.train.string_input_producer([tfrecord_path])
        reader = tf.TFRecordReader()

        # Take one example
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feats)

        # Extract individual features from example.
        image = tf.decode_raw(features["image_buffer"], tf.uint8)
        label = tf.cast(features["label"], tf.int32)
        height = tf.cast(features["image_height"], tf.int32)
        width = tf.cast(features["image_width"], tf.int32)
        channels = tf.cast(features["image_channels"], tf.int32)
        image = tf.reshape(image, tf.stack([height, width, channels]))
        
        # Start queue runners.
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Obtain numpy arrays for features.
        im, lab, w, h, c = sess.run([image, label, width, height, channels])

        # Perform some checks.
        Check(im.ndim == 3, "Invalid image dimensions count after reading from tfrecord.")
        Check(im.size == w * h * c, "Mismatch in image size after reading from tfrecord.")

        # Uncomment to see actual image.
        # cv2.imshow('image',im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Stop queue runners.
        coord.request_stop()
        coord.join(threads)

def write_tfrecord(in_dir, out_tfrecord_path):
    """
    Creates tfrecord at the given output path from the images in the given directory. Images need to be in the subfolders
    named after language codes.

    Args:
        in_dir :            Path to input directory with images.
        out_tfrecord_path : Path to output tfrecord file.
    """
    # Create tf record writer with the given output path.
    writer = tf.python_io.TFRecordWriter(out_tfrecord_path)

    # Input directory is expected to have subdirectories (named after language codes). Collect them.
    subdirs = next(os.walk(in_dir))[1]
    # Go over subdirectories.
    for subdir in subdirs:
        # Ensure that subdirectory name is language code.
        if subdir not in languages:
            raise Exception("Subdirectory name {} is not in supported list of languages".format(subdir))

        # Collect all image files within subdirectory.
        subdir_path = os.path.join(in_dir, subdir)
        png_files = os.listdir(subdir_path)
        # Go over image files.
        for pjg_file in png_files:
            png_file_path = os.path.join(subdir_path, pjg_file)
            # Read the image.
            img = cv2.imread(png_file_path)
            # Collect needed features that will be saved in tfrecord.
            buffer_string = img.tostring()
            width = img.shape[1]
            height = img.shape[0]
            channels = img.shape[2]
            label = language_to_output[subdir]
            # Create mapping from feature name to collected features.
            feats = {
                image_width_name : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                image_height_name : tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                image_channels_name : tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
                image_buffer_name : tf.train.Feature(bytes_list=tf.train.BytesList(value=[buffer_string])),
                label_name : tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
            # Create example with given features.
            example = tf.train.Example(features=tf.train.Features(feature=feats))

            # Save example to tfrecord.
            writer.write(example.SerializeToString())

    # Close the writer.
    writer.close()

if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', help='Path to input directory where samples are stored.', required=True)
    parser.add_argument('-out', help='Path to output tfrecord file.', required=True)
    args = parser.parse_args()
    args_dict = vars(args)
    # Write tfrecord.
    write_tfrecord(args_dict["in_dir"], args_dict["out"])
    # Check written tfrecord.
    check_tfrecord(args_dict["out"])