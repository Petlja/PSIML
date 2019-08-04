import data
from model import Model
from trainer import Trainer
import utils


def main():
    # Get dataset
    # Will be downloaded on the first run
    train_images, train_labels, valid_images, valid_labels = data.get_fashion_mnist()
    # Show some images from the dataset
    data.show_random_images(train_images, train_labels, 50)

    # Directory to save summaries to
    # From your conda environment run
    # tensorbard --logdir ../tf_playground/output
    # to see training details
    output = utils.get_output_dir()

    # Create model
    # TODO: Go to model.py file to implement model
    model = Model()

    # Lets train
    # TODO: Go to trainer.py file to implement trainer
    trainer = Trainer(train_images=train_images, train_labels=train_labels, valid_images=valid_images, valid_labels=valid_labels,
                      model=model, epochs=10, batch_size=10, output=output)
    trainer.train()

    trainer.save_final_accuracy()


if __name__ == "__main__":
    main()
