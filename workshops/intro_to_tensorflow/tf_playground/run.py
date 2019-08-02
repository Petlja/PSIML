import data
from model import Model
from trainer import Trainer
import utils


def main():
    # Generate and split data
    # Try and play with arguments (noise, val_factor)
    all_data = data.generate_data_spiral(numSamples=1000, noise=0.1)
    train_data, valid_data = data.split_data(all_data, val_factor=0.3)
    # Set show to True if you want to see generated dataset
    data.plot_data(train_data, valid_data, show=False)

    # Directory to save summaries to
    # From your conda environment run
    # tensorbard --logdir ../tf_playground/output
    # to see training details
    output = utils.get_output_dir()

    # Create model
    # Go to model.py file to make changes to the model
    model = Model()

    # Lets train
    # Try changing number of epochs and batch_size
    trainer = Trainer(train_data=train_data, valid_data=valid_data,
                      model=model, epochs=100, batch_size=10, output=output)
    trainer.train()

    trainer.save_final_accuracy()


if __name__ == "__main__":
    main()
