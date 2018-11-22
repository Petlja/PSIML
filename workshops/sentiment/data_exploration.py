import argparse
import pandas as pd
import logging
import collections
from utils import progress
import utils.io
import matplotlib.pyplot as plt
import data.sentiment140
from data.tokens import Tokenizer


def split(args):
    data.sentiment140.split_data(args.input, args.output)


def add_split_subparser(subparsers):
    subparser = subparsers.add_parser(name='split', help='Split sentiment140 training data to train, valid, test.',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser.add_argument('--input', required=True, help='sentiment 140 csv file with 1.6M tweets')
    subparser.add_argument('--output', required=True, help='output dir where train|valid|test.csv will be stored.')
    subparser.set_defaults(func=split)


def save_word_count(input_csv, output_json):
    data_frame = pd.read_csv(input_csv)
    tweets = data_frame.text.values
    word_count = collections.Counter()
    total = len(tweets) - 1
    tokenizer = Tokenizer()
    progress_bar = progress.ProgressBar(total=total, name="Count words")
    for iteration, tweet in enumerate(tweets):
        logging.debug('Tweet {0}'.format(tweet))
        for word in tokenizer(tweet):
            word_count[word] += 1
        progress_bar.show(iteration)
    utils.io.save_to_json(word_count, output_json, sort_keys=True, indent=2)


def count_words(args):
    save_word_count(args.input, args.output)


def add_count_words_subparser(subparsers):
    subparser = subparsers.add_parser(name='count_words', help='Extracts tokens and count',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser.add_argument('--input', required=True, help='Pandas csv frame with text and target')
    subparser.add_argument('--output', required=True, help='Output file in JSON format with word count')
    subparser.set_defaults(func=count_words)


def save_label_count(input_csv, output_json):
    data_frame = pd.read_csv(input_csv)
    tweet_sentiment = data_frame.target.values
    label_count = collections.Counter()
    total = len(tweet_sentiment) - 1
    progress_bar = progress.ProgressBar(total=total, name="Count labels")
    for iteration, label in enumerate(tweet_sentiment):
        label_count[str(label)] += 1
        progress_bar.show(iteration)
    utils.io.save_to_json(label_count, output_json, sort_keys=False, indent=2)


def count_labels(args):
    save_label_count(args.input, args.output)


def add_count_labels_subparser(subparsers):
    subparser = subparsers.add_parser(name='count_labels', help='Extracts tokens and count',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser.add_argument('--input', required=True, help='Pandas csv frame with text and target')
    subparser.add_argument('--output', required=True, help='Output file in JSON format with word count')
    subparser.set_defaults(func=count_labels)


def corpus_coverage(args):
    word_count = utils.io.load_from_json(args.input)
    logging.debug('Word count {0}'.format(len(word_count)))
    count_desc = sorted(list(word_count.values()), reverse=True)
    logging.debug('Count values {0}'.format(len(count_desc)))
    running = utils.math.running_sum(count_desc)
    total = sum(count_desc)
    xs = [x / total for x in running]
    logging.debug('xs {0}'.format(len(xs)))
    assert len(xs) == len(count_desc)
    indices = list(range(len(xs)))
    plt.plot(indices, xs)
    plt.xlabel('dictionary size')
    plt.ylabel('coverage')
    plt.show()
    xs = list(zip(count_desc, xs))
    logging.debug('xs {0}'.format(len(xs)))
    utils.io.save_to_json(xs, args.output, indent=2)


def add_coverage_subparser(subparsers):
    subparser = subparsers.add_parser(name='coverage', help='Gets text corpus coverage stats',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser.add_argument('--input', required=True, help='Word count JSON file')
    subparser.add_argument('--output', required=True, help='Output file in JSON format with coverage stats')
    subparser.set_defaults(func=corpus_coverage)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Manage tokens from tweets data set.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()
    add_split_subparser(subparsers)
    add_count_words_subparser(subparsers)
    add_count_labels_subparser(subparsers)
    add_coverage_subparser(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
