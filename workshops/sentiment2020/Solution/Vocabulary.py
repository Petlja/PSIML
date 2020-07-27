class Vocabulary:
    """The Vocabulary class maintains token to integer mapping needed for the rest of the machine learning pipeline"""

    def __init__(self, token_to_idx=None, add_unknown_token=True, unknown_token="<UNK>"):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
            add_unknown_token (bool): a flag that indicates whether to add the "unknown" token
            unknown_token (str): the "unknown" token to add into the vocabulary
        """
        # in case there is no pre-existing map specified, the map is empty
        if token_to_idx is None:
            token_to_idx = {}

        # set token to index map
        self._token_to_idx = token_to_idx

        # set index to token map
        self._idx_to_token = {
            idx: token for token, idx in self._token_to_idx.items()
        }

        # set a flag that indicates whether to add the "unknown" token
        self._add_unknown_token = add_unknown_token

        # set the "unknown" token
        self._unknown_token = unknown_token

        # set the index of the "unknown" token
        self.unknown_index = -1

        # add "unknown" token to the vocabulary, if the flag said so
        if (add_unknown_token):
            self.unknown_index = self.add_token(unknown_token)


    def to_serializable(self):
        """
        Returns a dictionary that can be serialized
        """
        return {
            "token_to_idx": self._token_to_idx,
            "add_unknown_token": self._add_unknown_token,
            "unknown_token": self._unknown_token,
        }


    @classmethod
    def from_serialiable(cls, dictionary):
        """
        Instantiates the Vocabulary from a serialized dictionary
        """
        return cls(**dictionary)


    def add_token(self, token):
        """
        Update mapping dictionaries to include the @token

        Args:
            token (str): the item to add into the Vocabulary
        Return:
            index (int): the integer corresponding to the token
        """
        # if the token is already in vocabulary
        if token in self._token_to_idx:
            # get index from the dictionary
            index = self._token_to_idx[token]
        else:
            # if the token is not in the Vocabulary, add it in the end of the dictionaries
            index = len(self._token_to_idx)
            # add the token in token to index map
            self._token_to_idx[token] = index
            # update index to token map
            self._idx_to_token[index] = token

        return index


    def find_token(self, token):
        """
        Retrieve the index associated with the token or index of the "unknown" token, if the token is not present in the Vocabulary

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            add_unknown_token must be set to true, to return the index of "unknown" token, if the token is not present in the Vocabulary
        """
        if self._add_unknown_token:
            # returns the index of the token if the token is present in the dictionary or index of the "unknown", otherwise
            return self._token_to_idx.get(token, self.unknown_index)
        else:
            # return the index of the token
            return self._token_to_idx[token]


    def find_index(self, index):
        """
        Return token associated with the index

        Args:
            index (str): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not present in the Vocabulary
        """
        # if the index is not present in the dictionary, raise the Key Error
        if index not in self._idx_to_token:
            raise KeyError("The index (%d) is not in the Vocabulary" % index)

        # return the token correspoding to the index
        return self._idx_to_token[index]


    def __str__(self):
        """Returns the string describing the Vocabulary"""
        return "Vocabulary(size=%d)" % len(self)


    def __len__(self):
        """Returns the length of the Vocabulary"""
        return len(self._token_to_idx)
