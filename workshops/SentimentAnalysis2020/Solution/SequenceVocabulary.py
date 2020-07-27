from Vocabulary import Vocabulary

class SequenceVocabulary(Vocabulary):
    """
    SequenceVocabulary is a subclass of the stanard Vocabulary class, that bundles four special tokens that are commonly used for sequence data:
        1. the UNK token - token representing the unknown word
        2. the MASK token - token representing padding for sequences of variable length
        3. the BEGINNING-OF-SEQUENCE token - token representing the beginning of the sequence
        4. the END-OF-SEQUENCE token - token representing the end of the sequence

    The rest of the functionalities of the Vocabulary class stays unchanged
    """
    def __init__(self, token_to_idx=None):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
        """
        # construct the base Vocabulary
        # add_unknown_token is initally set to false, as this token will be added a bit later
        super(SequenceVocabulary, self).__init__(token_to_idx=token_to_idx, add_unknown_token=False)

        # firstly add mask token to the Vocabulary and store its index
        self.mask_index = super().add_token("<MASK>")

        # add the unknown token to the Vocabulary and store its index
        self.unknown_index = super().add_token("<UNK>")
        # set the flag that the unknown token is used, so to it can be returned in cases when the word is not present in the Vocabulary
        self._add_unknown_token = True

        # add the beginning-of-sequence token to teh Vocabulary and store its index
        self.begin_seq_index = super().add_token("<BEGINNING-OF-SEQUENCE>")

        # add the end-of-sequence token to teh Vocabulary and store its index
        self.end_seq_index = super().add_token("<END-OF-SEQUENCE>")
