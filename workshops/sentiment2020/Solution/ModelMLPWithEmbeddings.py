import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentClassifierMLPWithEmbeddings(nn.Module):
    """A 2-layer multilayer perceptron based classifier that uses word embeddings as an input sequence representation"""

    def __init__(self, embedding_size, num_embeddings, hidden_dim, output_dim, pretrained_embedding_matrix=None, padding_idx=0):
        """
        Args:
            embedding_size (int): the size of the embedding vector
            num_embeddings (int): the number of words to embed
            hidden_dim (int): the size of the hidden layer
            output_dim (int): the size of the prediction vector
            pretrained_embedding_matrix (numpy.array): previously trained word embeddings
            padding_idx (int): an index in the Vocabulary representing the <MASK> token (padding)
        """
        # call the base initialization
        super(SentimentClassifierMLPWithEmbeddings, self).__init__()

        # Define the model

        if pretrained_embedding_matrix is None:
            # instantiate the Embedding layer without initial weights
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_size,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx
            )
        else:
            # firstly convert pretrained_embedding_matrix into tensor
            pretrained_embedding_matrix = torch.from_numpy(pretrained_embedding_matrix).float()

            # instantiate the Embedding layer with initial weights
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_size,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx,
                _weight=pretrained_embedding_matrix
            )

        # first linear layer with number of inputs correspoding to the embedding size and number of outputs correspoding to the hidden state size
        self.fc1 = nn.Linear(in_features=embedding_size, out_features=hidden_dim)

        # second linear layer with number of inputs correspoding to the hidden state size and number of outputs corresponding the number of output classes
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)


    def forward(self, x_in, apply_softmax=False):
        """
        The forward pass of the Classifier

        Args:
            x_in (torch.Tensor): an input data tensor with input shape (batch, dataset._max_sequence_length)
            apply_softmax (bool): a flag for the softmax activation, which should be false if used with the cross-entropy losses
        Returns:
            resulting tensor, with shape (batch, output_dim)
        """
        # TODO workshop task
        # Steps:
        # 1. create vectors for each word in the input data tensor, by converting the indices to vectors
        # 2. combine the vector in some way such that it captures the overall context of the sequence
        # 3. calculate the output of the first linear layer
        # 4. apply non-linear function to the output of the linear layer
        # 5. calculate the output of the second linear layer
        # 6. apply softmax function to the calculate output, if needed
        # 7. return output

        # create vectors for each word in the input data tensor, by converting the indices to vectors
        x_embedded = self.embeddings(x_in.long())

        # combine the vector in some way such that it captures the overall context of the sequence (e.g. sum the vectors for all the words)
        x_embedded_sum = x_embedded.sum(dim=1)

        # calculate the output of the first linear layer
        y_out = self.fc1(x_embedded_sum)

        # apply non-linear function to the output of the linear layer
        y_out = F.relu(y_out)

        # calculate the output of the second linear layer
        y_out = self.fc2(y_out)

        # apply softmax function to the calculate output, if needed
        if (apply_softmax):
            y_out = F.softmax(y_out, dim=1)

        return y_out
        # END workshop task
