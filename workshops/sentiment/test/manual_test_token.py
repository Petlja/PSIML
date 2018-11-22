import logging
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data import tokens


def test(token_count_json):
    tokenizer = tokens.WordDictionary(word_count_file=token_count_json)
    text = 'This is a tweet #tweet with at least one out-of-dictionary word!'
    print(text)
    token_ids = tokenizer.word_ids(text)
    print(' '.join([str(x) for x in token_ids]))
    text_back = tokenizer.text(token_ids)
    print(text_back)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Simple tokenization smoke test',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', required=True, help='Token count JSON file')
    args = parser.parse_args()
    test(token_count_json=args.input)
