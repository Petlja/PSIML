import model
import model_utils
import runner
import trainer
from data_utils import download_cifar10, load_cifar10, train_val_split
from vis_utils import visualize_dataset_sample, visualize_classification

import os
import sys
import numpy as np

def prepare_dataset():
    """
    Prepares CIFAR-10 dataset.
    - 50,000 development images to be used for building the model.
    - 10,000 test images to be used for evaluating the final model.
    - Images are 32 x 32 pixels in size, with 3 channels (R, G, and B, in that order).
    - Each image is labeled with one of 10 classes.
    """
    # Download data to local storage.
    download_cifar10()

    # Load data from file into memory.
    class_names, images_dev, labels_dev, images_test, labels_test = load_cifar10()

    # Visualize some random examples from each class.
    visualize_dataset_sample(images_dev, labels_dev, sample_size=10)

def evaluate_image(load_checkpoint):
    """
    Creates a convolutional network, optionally loads its weights from file, and
    runs it on a random test image from CIFAR-10. Visualizes top predictions.

    Args:
        load_checkpoint: Boolean flag indicating if weights should be loaded from file.
    """
    # Load data from file into memory.
    class_names, _, _, images_test, labels_test = load_cifar10()

    # Create network.
    my_model = model.Model()

    # Create runner, optionally load a weights from file.
    my_runner = runner.Runner(model=my_model)
    if load_checkpoint:
        my_runner.load(os.path.join(".", "checkpoints", "my_model"))

    # Evaluate network on a random test image.
    image_index = np.random.randint(0, images_test.shape[0])
    image = images_test[image_index]
    label = labels_test[image_index]
    guess_class, guess_prob = my_runner.run(image)

    # Visualize the result.
    visualize_classification(image, label, guess_class, guess_prob, class_names)

def evaluate_dataset(load_checkpoint):
    """
    Creates a convolutional network, optionally loads its weights from file, and
    runs it on the whole CIFAR-10 test set. Calculates loss and accuracy on the
    test set.

    Args:
        load_checkpoint: Boolean flag indicating if weights should be loaded from file.
    """
    # Load data from file into memory.
    class_names, _, _, images_test, labels_test = load_cifar10()

    # Create network.
    my_model = model.Model()

    # Create trainer, optionally load a weights from file.
    my_trainer = trainer.Trainer(model=my_model, learning_rate=5e-4)
    if load_checkpoint:
        my_runner.load(os.path.join(".", "checkpoints", "my_model"))

    # Evaluate network on test set.
    my_trainer.evaluate(images_test, labels_test, batch_size=64, print_every=100)

def train():
    """
    Trains a convolutional network on CIFAR-10 for one epoch, and saves the resulting model.
    """
    # Load data from file into memory.
    _, images_dev, labels_dev, images_test, labels_test = load_cifar10()

    # Take a random 10% of development data for validation. Use the rest for training.
    # Randomization is used to achieve similar distributions for training and validation data.
    images_train, labels_train, images_val, labels_val = train_val_split(images_dev, labels_dev, val_fraction=0.1)

    # Create network.
    my_model = model.Model()

    # Train network for 5 epochs.
    my_trainer = trainer.Trainer(model=my_model, learning_rate=5e-4)
    my_trainer.train(images_train, labels_train, images_val, labels_val, batch_size=64, epochs=5, print_every=100)

    # Evaluate network on test set.
    my_trainer.evaluate(images_test, labels_test, batch_size=64, print_every=100)

    # Save the resulting model.
    my_trainer.save(os.path.join(".", "checkpoints", "my_model"))

def print_usage():
    print("Usage: %s [1 | 2 | 3 | 4 | 5 | 6 | 7]" % os.path.basename(__file__))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
    else:
        scenario = int(sys.argv[1])
        if scenario == 1:
            prepare_dataset()
        elif scenario == 2:
            evaluate_image(load_checkpoint=False)
        elif scenario == 3:
            evaluate_dataset(load_checkpoint=False)
        elif scenario in [4, 6]:
            train()
        elif scenario in [5, 7]:
            evaluate_image(load_checkpoint=True)
        else:
            print_usage()
