import argparse
import logging
import model
import tfrecord_reader


def train(args):
    train_tfrecord_path = args.train
    valid_tfrecord_path = args.valid
    output = args.output
    logging.info('Training')
    trainer = model.Trainer(train_tfrecord_path=train_tfrecord_path, valid_tfrecord_path=valid_tfrecord_path, output=output)
    trainer.train()

def analyze(args):
    checkpoint = args.model
    dataset_path = args.dataset
    output = args.output
    logging.info('Analyzing')
    runner = model.Runner(checkpoint=checkpoint, dataset_path=dataset_path, output=output)
    runner.analyze()

def add_train_subparser(subparsers):
    subparser = subparsers.add_parser(name='train', help='Trains the model',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser.add_argument('--train', required=True, help='Root folder containing train images')
    subparser.add_argument('--valid', required=True, help='Root folder containing valid images')
    subparser.add_argument('--output', required=True, help='Output directory')
    subparser.set_defaults(func=train)

def add_analyze_subparser(subparsers):
    subparser = subparsers.add_parser(name='analyze', help='Analyzes the model',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser.add_argument('--model', required=True, help='Path to file containing model snapshot')
    subparser.add_argument('--dataset', required=True, help='Root folder containing images to use for analysis')
    subparser.add_argument('--output', required=True, help='Output directory')
    subparser.set_defaults(func=analyze)

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Trains and analyzes script detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()
    add_train_subparser(subparsers)
    add_analyze_subparser(subparsers)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
