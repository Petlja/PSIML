import argparse
import logging
import model
import image_reader


def train(args):
    train_root = args.train
    valid_root = args.valid
    output = args.output
    line_height = 32
    logging.info('Reading validation set')
    valid_data = image_reader.ImageReader(line_height=line_height, root_dir=valid_root)
    logging.info('Reading training set')
    train_data = image_reader.ImageReader(line_height=line_height, root_dir=train_root)
    logging.info('Training')
    trainer = model.Trainer(train_data=train_data, valid_data=valid_data, output=output)
    trainer.train()


def add_train_subparser(subparsers):
    subparser = subparsers.add_parser(name='train', help='Trains the model',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser.add_argument('--train', required=True, help='Root folder containing train images')
    subparser.add_argument('--valid', required=True, help='Root folder containing valid images')
    subparser.add_argument('--output', required=True, help='Output directory')
    subparser.set_defaults(func=train)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Trains script detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()
    add_train_subparser(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
