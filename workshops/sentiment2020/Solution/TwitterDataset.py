import pandas as pd
import torch
from torch.utils.data import Dataset

from TwitterOneHotVectorizer import TwitterOneHotVectorizer
from TwitterSequenceVectorizer import TwitterSequenceVectorizer


class TwitterDataset(Dataset):
    def __init__(self, dataset_df, vectorizer):
        """
        Args:
            dataset_df (pandas.DataFrame): the data frame containing the preprocessed dataset
            vectorizer (TweetVectorizer): vectorizer instantiated from this dataset
        """
        # set up the dataset
        self.dataset_df = dataset_df
        # set up the vectorizer
        self._vectorizer = vectorizer

        # set up training dataset
        self.train_df = self.dataset_df[self.dataset_df.split == "train"]
        # set up the size of the training dataset
        self.train_size = len(self.train_df)

        # set up validation dataset
        self.validation_df = self.dataset_df[self.dataset_df.split == "validation"]
        # set up the size of the validation dataset
        self.validation_size = len(self.validation_df)

        # set up test dataset
        self.test_df = self.dataset_df[self.dataset_df.split == "test"]
        # set up the size of the test dataset
        self.test_size = len(self.test_df)

        # set look up dictionary
        self._split_dictionary = {
            "train": (self.train_df, self.train_size),
            "validation": (self.validation_df, self.validation_size),
            "test": (self.test_df, self.test_size),
        }

        # set currently chosen set
        self.set_split("train")

        # for sequence representations only - find the maximal sequence length in the training dataset
        self._max_sequence_length = self.train_df.text.apply(len).max()


    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv_path, representation="one-hot"):
        """
        Load dataset and make a new vectorizer from scratch

        Args:
            dataset_csv_path (str): path to the dataset
            representation (str): representation of the text sequence, options are: "one-hot" and "indices"
        Returns:
            an instance of the TwitterDataset
        """
        # load the dataset
        dataset_df = pd.read_csv(dataset_csv_path)

        # make a new vectorizer based on the chosen text representation
        if representation == "one-hot":
            vectorizer = TwitterOneHotVectorizer.from_dataframe(dataset_df)
        elif representation == "indices":
            vectorizer = TwitterSequenceVectorizer.from_dataframe(dataset_df)
        else:
            raise Exception("Represention not supporeted")

        return cls(dataset_df, vectorizer)


    def get_vectorizer(self):
        """Returns the vectorizer"""
        return self._vectorizer


    def set_split(self, split="train"):
        """
        Selects the currently chosen set: train, validation or test

        Args:
            split (str): one of "train", "validation" or "test"
        """
        # sets the current split
        self._current_split = split
        # sets the current dataset and its size
        self._current_df, self._curret_size = self._split_dictionary[split]


    def __len__(self):
        """
        Returns the size of the currently chosen dataset split

        Note: This method is defined in abstract Dataset class and must be implemented in its inherited class
        """
        return self._curret_size


    def __getitem__(self, index):
        """
        The primary entry point method for PyTorch datasets

        Note: This method is defined in abstract Dataset class and must be implemented in its inherited class

        Args:
            index (int): the index of the data point
        Return:
            a dict of the data point's features (x_data) and label (y_target)
        """
        # get row for the data point
        row = self._current_df.iloc[index]

        # vectorize the text of the tweets
        text_vector = self._vectorizer.vectorize(row.text, vector_length = self._max_sequence_length)

        # target index
        target_index = self._vectorizer.target_vocabulary.find_token(row.target)

        return {
            "x_data": text_vector,
            "y_target": target_index,
        }


    def get_num_batches(self, batch_size):
        """
        Given a batch size, returns the number of batches in teh current dataset

        Args:
            batch_size (int) : the batch size
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size
