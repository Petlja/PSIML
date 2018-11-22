import argparse
import logging
import data.sentiment140
import model
from data.tokens import WordDictionary


def train(args):
    train_data_file = args.train
    valid_data_file = args.valid
    word_count_file = args.word_count
    output = args.output
    word_dictionary = WordDictionary(word_count_file=word_count_file)
    valid_data = data.sentiment140.TweetData(valid_data_file, word_dictionary=word_dictionary)
    train_data = data.sentiment140.TweetData(train_data_file, word_dictionary=word_dictionary)
    trainer = model.Trainer(train_data=train_data, valid_data=valid_data, word_dictionary=word_dictionary,
                            output=output)
    trainer.train()


def add_train_subparser(subparsers):
    subparser = subparsers.add_parser(name='train', help='Trains the model',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser.add_argument('--train', required=True, help='CSV training data frame')
    subparser.add_argument('--valid', required=True, help='CSV validation data frame')
    subparser.add_argument('--word_count', required=True, help='JSON file with word count')
    subparser.add_argument('--output', required=True, help='Output directory')
    subparser.set_defaults(func=train)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Trains sentiment analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()
    add_train_subparser(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
