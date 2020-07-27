import torch
import torch.nn as nn
import torch.nn.functional as F

class ElmanRNN(nn.Module):
    """An Elman RNN module built using RNNCell"""

    def __init__(self, input_size, hidden_size, batch_first=False):
        """
        Args:
            input_dim (int): the size of the input feature vector
            hidden_size (int): the size of the hidden state vectors
            batch_first (bool): flag whether the batch is the 0th dimension in the input tensor

        """
        # call the base initialization
        super(ElmanRNN, self).__init__()

        # Define the model
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)

        # store the rest of the parameters
        self.batch_first = batch_first
        self.hidden_size = hidden_size


    def _initialize_hidden(self, batch_size):
        """Initialize the hidden state vector for the batch with zeros"""
        return torch.zeros((batch_size, self.hidden_size))


    def forward(self, x_in, initial_hidden=None):
        """
        The forward pass of the ElmanRNN

        Args:
            x_in (torch.Tensor): an input data tensor
                if self.batch_first is true, the input shape is (batch_size, sequence_size, features_size)
                if self.batch_first is false, the input shape is (sequence_size, batch_size, features_size)
            initial_hidden (torch.Tensor): the initial hidden state for the rnn
        Returns:
            hiddens (torch.Tensor): the outputs of the rnn for each element in the sequence (time-stamp)
                if self.batch_first is true, the output shape is (batch_size, sequence_size, hidden_size)
                if self.batch_first is false, the input shape is (sequence_size, batch_size, hidden_size)
        """
        # firstly convert the input tensor in the expected format, the rnn_cell expects it in shape (sequence_size, batch_size, features_size)
        if self.batch_first:
            batch_size, sequence_size, _ = x_in.size()
            # change the order of the 0th and 1st dimension
            x_in = x_in.permute(1, 0, 2)
        else:
            sequence_size, batch_size, _ = x_in.size()

        # set-up the initial hidden state, if not provided in advance
        if initial_hidden is None:
            initial_hidden = self._initialize_hidden(batch_size)
            initial_hidden = initial_hidden.to(x_in.device)

        # initialize the list of hidden states
        hiddens = []

        # set-up the initial hidden state
        hidden_t = initial_hidden

        # TODO workshop task
        # iterate through the sequence and calculate new hidden cell
        # steps for each sequence:
        # 1. Each new hidden state is calculated using current input element and previous hidden state. rnn_cell encapsulates the computation of the new hidden state
        # 2. Add calculated state to the list
        
        # END workshop task

        # the list is converted to Tensor, which shape (sequence_size, batch_size, hidden_size)
        hiddens = torch.stack(hiddens)

        # if the batch was the 0th dimension in the input, permute the hiddens Tensor to have this in the output as well
        if self.batch_first:
            # change the order of the 0th and 1st dimension
            hiddens = hiddens.permute(1, 0, 2)

        return hiddens
