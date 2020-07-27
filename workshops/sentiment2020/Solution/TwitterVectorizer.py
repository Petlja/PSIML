import numpy as np
import string
from collections import Counter

from nltk.tokenize import TweetTokenizer

from Constants import c_frequencyCutoff
from Vocabulary import Vocabulary

class TwitterVectorizer:
    """The Vectorizer that converts text to numberic vectors"""

    def __init__(self, text_vocabulary, target_vocabulary):
        """
        Args:
            text_vocabulary (Vocabulary): maps words to integers
            target_vocabulary (Vocabulary): maps class labels to integers
        """
        # set up the Vocabulary for text column
        self.text_vocabulary = text_vocabulary

        # set up the Vocabulary for target column
        self.target_vocabulary = target_vocabulary

        # initialize TweetTokenizer
        self._tokenizer = TweetTokenizer()


    def vectorize(self, text, vector_length):
        """
        Create a vector representation for the text

        Args:
            text (str): text of the tweet
            vector_length (int): length of the resulting vector
        Returns:
            vector representation (np.ndarray): a vector representation for the text
        """
        # this is an abstract method, concrete implentations are provided in the subclasses
        pass


    @classmethod
    def _get_text_vocabulary(cls):
        """
        Returns the Vocabulary that should be used for the text column
        """
        # this is an abstract method, concrete implentations are provided in the subclasses
        pass


    @classmethod
    def from_dataframe(cls, dataset_df, cutoff=c_frequencyCutoff):
        """
        Instantiate the Vectorizer from the dataset dataframe

        Args:
            dataset_df (pandas.DataFrame): the tweets dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the TwitterVectorizer
        """
        # instantiate the Vocabulary for text column
        text_vocabulary = cls._get_text_vocabulary()

        # instantiate the Vocabulary for target column
        target_vocabulary = Vocabulary(add_unknown_token=False)

        # add elements to Target Vocabulary
        for target in sorted(set(dataset_df.target)):
            target_vocabulary.add_token(target)

        # Tweet Tokenizer to split text into tokens
        tokenizer = TweetTokenizer()

        # add word to the Text Vocabulary, if its frequency > cutoff
        word_counts = Counter()

        # iterate through the dataset
        for text in dataset_df.text:
            # split text into words
            words = tokenizer.tokenize(text)

            # update word_counts for all words in the text
            for word in words:
                word_counts[word] += 1

        # for all extacted words
        for word, count in word_counts.items():
            # if the word is not punctuation and it appears more than @cutoff times, add it to the Vocabulary
            if (word not in string.punctuation) and (count > cutoff):
                # add token to the Vocabulary
                text_vocabulary.add_token(word)

        return cls(text_vocabulary, target_vocabulary)


    @classmethod
    def from_serializable(cls, contents):
        """
        Intantiate a TwitterVectorizer from serializable dictionary

        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the TwitterVectorizer
        """
        # load the Text Vocabulary
        text_vocabulary = Vocabulary.from_serialiable(contents["text_vocabulary"])

        # load the Target Vocabulary
        target_vocabulary = Vocabulary.from_serialiable(contents["target_vocabulary"])

        return cls(text_vocabulary=text_vocabulary, target_vocabulary=target_vocabulary)


    def to_serializable(self):
        """
        Create the serializable dictionary

        Returns:
            Contents (dict): the serializable dictionary
        """
        return {
            "text_vocabulary": self.text_vocabulary.to_serializable(),
            "target_vocabulary": self.target_vocabulary.to_serializable(),
        }
