import numpy as np
import string

from Vocabulary import Vocabulary
from TwitterVectorizer import TwitterVectorizer

class TwitterOneHotVectorizer(TwitterVectorizer):
    """The Vectorizer that converts text to numberic vectors"""

    def __init__(self, text_vocabulary, target_vocabulary):
        """
        Args:
            text_vocabulary (Vocabulary): maps words to integers
            target_vocabulary (Vocabulary): maps class labels to integers
        """
        # construct the base TwitterVectorizer
        super(TwitterOneHotVectorizer, self).__init__(text_vocabulary, target_vocabulary)


    def vectorize(self, text, vector_length=-1):
        """
        Create a one-hot vector for the text

        Args:
            text (str): text of the tweet
            vector_length (int): length of the resulting vector
        Returns:
            one_hot encoding (np.ndarray): the collapsed one-hot encoding
        """
        # initialize the one-hot encoding vector
        one_hot = np.zeros(len(self.text_vocabulary), dtype=np.float32)

        # split text into tokens
        tokens = self._tokenizer.tokenize(text)

        for token in tokens:
            # in case the token in not a punctuation
            if token not in string.punctuation:
                # find index of the token in the Vocabulary
                index = self.text_vocabulary.find_token(token)
                # set-up the flag in one-hot encoding
                one_hot[index] = 1

        return one_hot


    @classmethod
    def _get_text_vocabulary(cls):
        """Returns the Vocabulary that should be used for the text column"""
        return Vocabulary(add_unknown_token=True)
