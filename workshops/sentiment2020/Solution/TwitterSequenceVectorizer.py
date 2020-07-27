import numpy as np
import string

from SequenceVocabulary import SequenceVocabulary
from TwitterVectorizer import TwitterVectorizer


class TwitterSequenceVectorizer(TwitterVectorizer):
    """The Vectorizer that converts text to numberic vectors"""

    def __init__(self, text_vocabulary, target_vocabulary):
        """
        Args:
            text_vocabulary (Vocabulary): maps words to integers
            target_vocabulary (Vocabulary): maps class labels to integers
        """
        # construct the base TwitterVectorizer
        super(TwitterSequenceVectorizer, self).__init__(text_vocabulary, target_vocabulary)


    def vectorize(self, text, vector_length=-1):
        """
        Create a vector for the text

        Args:
            text (str): text of the tweet
            vector_length (int): an argument that forces the length of the output vector
        Returns:
            out_vector (np.ndarray): the vectorized tweet
        """
        # split text into tokens
        tokens = self._tokenizer.tokenize(text)

        # TODO workshop task
        # Steps:
        # 1. add the BEGINNING-OF-SEQUENCE token to the indices vector
        # 2. in case the token in not a punctuation, find index of the token in the Vocabulary and add it to vector
        # 3. add the END-OF_SEQUENCE token to the vector
        # 4. copy the indices vector to the output vector
        # 5. pad the output vector sequence with MASK tokens, to fill the remaining "free" space in the output vector of fixed size
        # 6. return output vector

        # END workshop task


    @classmethod
    def _get_text_vocabulary(cls):
        """Returns the Vocabulary that should be used for the text column"""
        return SequenceVocabulary()
