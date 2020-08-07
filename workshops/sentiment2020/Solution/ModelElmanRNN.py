import torch
import torch.nn as nn
import torch.nn.functional as F

from ElmanRNN import ElmanRNN

class SentimentClassifierElmanRNN(nn.Module):
    """A RNN to extract feature representation and 2-layer multilayer perceptron to do the classification"""

    def __init__(self, embedding_size, num_embeddings, rnn_hidden_dim, output_dim, pretrained_embedding_matrix=None, padding_idx=0, batch_first=True):
        """
        Args:
            embedding_size (int): the size of the embedding vector
            num_embeddings (int): the number of words to embed
            rnn_hidden_dim (int): the size of the RNN's hidden state
            output_dim (int): the size of the prediction vector
            pretrained_embedding_matrix (numpy.array): previously trained word embeddings
            padding_idx (int): an index in the Vocabulary representing the <MASK> token (padding)

        """
        # call the base initialization
        super(SentimentClassifierElmanRNN, self).__init__()

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


        # rnn model with number of inputs correspoding to the embedding size and defined size of the hidden state
        self.rnn = ElmanRNN(input_size=embedding_size, hidden_size=rnn_hidden_dim, batch_first=batch_first)

        # first linear layer with number of inputs correspoding to the size of the rnn's hidden state feature vector and number of outputs being equal to the number of inputs
        self.fc1 = nn.Linear(in_features=rnn_hidden_dim, out_features=rnn_hidden_dim)

        # second linear layer with number of inputs correspoding to the hidden state size and number of outputs corresponding the number of output classes
        self.fc2 = nn.Linear(in_features=rnn_hidden_dim, out_features=output_dim)


    def forward(self, x_in, x_lengths=None, apply_softmax=False):
        """
        The forward pass of the Classifier

        Args:
            x_in (torch.Tensor): an input data tensor with input shape (batch, dataset._max_sequence_length)
            x_length (torch.Tensor): the lengths of each sequence in the batch
            apply_softmax (bool): a flag for the softmax activation, which should be false if used with the cross-entropy losses
        Returns:
            resulting tensor, with shape (batch, output_dim)
        """
        # TODO workshop task
        # Steps:
        # 1. Create vectors for each word in the input data tensor, by converting the indices to vectors
        # 2. Create rnn hidden state vectors for each word in the input data tensor
        # 3. Take the hidden state vector of the last word in the sequence and if
        # 3.1. if the original sequence length is specified, find the hidden state that corresponds with the last word in the sequence
        # 3.2. if the original sequence length is not specified, it is assumed that the last hidden state corresponds with the last word in the sequence
        # 4. calculate the output of the first linear layer
        # 5. apply non-linear function to the output of the linear layer
        # 6. calculate the output of the second linear layer
        # 7. apply softmax function to the calculate output, if needed
        # 8. return output

        # create vectors for each word in the input data tensor, by converting the indices to vectors
        x_embedded = self.embeddings(x_in.long())

        # create rnn hidden state vectors for each word in the input data tensor
        y_out = self.rnn(x_embedded)

        # take the hidden state vector of the last word in the sequence
        if x_lengths is not None:
            # if the original sequence length is specified, find the hidden state that corresponds with the last word in the sequence
            y_out = self._column_gether(y_out, x_lengths)
        else:
            # if the original sequence length is not specified, it is assumed that the last hidden state corresponds with the last word in the sequence
            y_out = y_out[:, -1, :]

        # calculate the output of the first linear layer
        y_out = self.fc1(y_out)

        # apply non-linear function to the output of the linear layer
        y_out = F.relu(y_out)

        # calculate the output of the second linear layer
        y_out = self.fc2(y_out)

        # apply softmax function to the calculate output, if needed
        if (apply_softmax):
            y_out = F.softmax(y_out, dim=1)

        return y_out
        # END workshop task


    def _column_gether(self, y_out, x_lengths):
        """
        Get a specific vector from each batch data point in y_out

        Args:
            y_out (torch.FloatTensor): hidden states for each element in the sequence, with shape (batch_size, sequence_size, feature_size)
            x_lengths (torch.FloatTensor): lengths of each sequence in the batch, with shape (batch_size, 1)
        Returns:
            out (torch.FloatTensor): hidden state of the last element in the sequence, with shape (batch_size, feature_size)
        """
        x_lengths = x_lengths.long().detach().cpu().numpy() - 1

        out = []
        for batch_index, column_index in enumerate(x_lengths):
            out.append(y_out[batch_index, column_index])

        return torch.stack(out)
