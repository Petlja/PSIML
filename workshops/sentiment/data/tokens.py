from nltk.tokenize import TweetTokenizer
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils.io
import utils.math


UNKNOWN_WORD = '~unknown~'
EOS_WORD = '~end~'
PADDING_WORD = '~padding~'


class Tokenizer(object):
    def __init__(self):
        self.tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)

    def __call__(self, text):
        return self.tokenizer.tokenize(text)


def _percentile(percentile_values, target_percentile):
    for i, percentile in enumerate(percentile_values):
        if percentile >= target_percentile:
            return i
    return len(percentile_values) - 1


class WordDictionary(object):
    def __init__(self, word_count_file, percentile=0.95):
        word_count = utils.io.load_from_json(word_count_file)
        count_desc = sorted(list(word_count.values()), reverse=True)
        running_sum = utils.math.running_sum(count_desc)
        total = sum(count_desc)
        percentiles = [x / total for x in running_sum]
        index = _percentile(percentiles, percentile)
        min_count = count_desc[index]
        self._word2id = {}
        self._id2word = {}
        self._add_word(PADDING_WORD)
        words = sorted(list(word_count.keys()))
        for word in words:
            if word_count[word] >= min_count:
                self._add_word(word)
        self._add_word(EOS_WORD)
        self._add_word(UNKNOWN_WORD)
        self._tokenizer = Tokenizer()
        logging.debug("Dictionary size: {0}".format(self.dictionary_size()))

    def _add_word(self, word):
        assert len(self._id2word) == len(self._word2id)
        word_id = len(self._word2id)
        self._id2word[word_id] = word
        self._word2id[word] = word_id

    def dictionary_size(self):
        return len(self._word2id)

    def word_id(self, word):
        if word in self._word2id:
            return self._word2id[word]
        else:
            return self._word2id[UNKNOWN_WORD]

    def word_ids(self, text):
        words = self._tokenizer(text)
        words.append(EOS_WORD)
        return [self.word_id(word) for word in words]

    def word(self, word_id):
        return self._id2word[word_id]

    def text(self, word_ids):
        words = [self.word(token_id) for token_id in word_ids]
        return ' '.join(words)
