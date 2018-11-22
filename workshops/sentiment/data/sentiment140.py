import pandas as pd
import os
import sys
import logging
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils.io

# Sentiment140 column names
_TEXT = 'text'
_TARGET = 'target'
_ID = 'id'
_DATE = 'date'
_FLAG = 'flag'
_USER = 'user'


def _read_sentiment140_csv(data_file):
    return pd.read_csv(data_file, encoding='cp1252',
                       names=[_TARGET, _ID, _DATE, _FLAG, _USER, _TEXT])


def _split_users(users):
    """
    Splits users to train, validation and test sets.
    """
    user_count = len(users)
    train_count = int(user_count * 0.8)
    valid_count = int(user_count * 0.1)
    train_users = users[:train_count]
    valid_users = users[train_count:train_count + valid_count]
    test_users = users[train_count + valid_count:]
    return train_users, valid_users, test_users


def _select_users_data(data_frame, users):
    """
    Selects text and target sentiment for subset of users from data corpus.
    """
    selected = data_frame.loc[data_frame.user.isin(users)][[_TEXT, _TARGET]]
    selected.target = selected.target / 2
    selected.target = selected.target.astype(int)
    return selected


def _save_text_and_target(data_frame, output_file):
    """
    Saves data frame in csv format.
    """
    utils.io.ensure_parent_exists(output_file)
    data_frame.to_csv(output_file)


def _load_text_and_target(input_file, encoding='cp1252'):
    """
    Loads pandas data frame with text and target sentiment columns.
    :param input_file: pandas data frame in csv format produced by splitting original sentiment140.csv.
    :param encoding: File encoding
    :return: Pandas data frame with text and target sentiment columns.
    """
    return pd.read_csv(input_file, encoding=encoding, names=[_TEXT, _TARGET])


def split_data(data_file, output_dir):
    """
    Splits data from original sentiment140.csv into train, validation and test sets.
    Splitting is done such that tweets from the same user will be stored in only one of the data splits.
    :param data_file: Original sentiment140.csv
    :param output_dir: Output directory where train.csv, test.csv and valid.csv will be created
    :return: None
    """
    data = _read_sentiment140_csv(data_file)
    logging.info("Found {0} tweets".format(len(data)))
    users = list(set(data.user.values))
    logging.info("Found {0} users".format(len(users)))
    train_users, valid_users, test_users = _split_users(users)
    logging.info(
        "User split: train {0}; valid {1}; test {2}".format(len(train_users), len(valid_users), len(test_users)))

    train = _select_users_data(data, train_users)
    logging.info("Train tweets: {0}".format(len(train)))
    output_file = os.path.join(output_dir, 'train.csv')
    _save_text_and_target(train, output_file)

    valid = _select_users_data(data, valid_users)
    logging.info("Valid tweets: {0}".format(len(valid)))
    output_file = os.path.join(output_dir, 'valid.csv')
    _save_text_and_target(valid, output_file)

    test = _select_users_data(data, test_users)
    logging.info("Test tweets: {0}".format(len(test)))
    output_file = os.path.join(output_dir, 'test.csv')
    _save_text_and_target(test, output_file)


class TweetData(object):
    def __init__(self, data_frame_file, word_dictionary):
        self._data = _load_text_and_target(data_frame_file, encoding='utf-8')
        self._tweets = self._data.text.values
        self._labels = self._data.target.values
        assert len(self._tweets) == len(self._labels)
        self._indices = list(range(self.__len__()))
        self._dictionary = word_dictionary

    def shuffle(self):
        random.shuffle(self._indices)

    def __len__(self):
        return len(self._tweets) - 1

    def __getitem__(self, index):
        index = self._indices[index]
        index += 1
        tweet = self._tweets[index]
        word_ids = self._dictionary.word_ids(tweet)
        word_ids = np.array(word_ids)
        return word_ids, int(self._labels[index])
