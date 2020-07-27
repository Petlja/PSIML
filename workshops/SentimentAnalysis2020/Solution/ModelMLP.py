import torch.nn as nn
import torch.nn.functional as F

class SentimentClassifierMLP(nn.Module):
    """A 2-layer multilayer perceptron based classifier that uses one-hot encoding as an input sequence representation"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): the size of the input feature vector
            hidden_dim (int): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
        """
        # call the base initialization
        super(SentimentClassifierMLP, self).__init__()

        # Define the model

        # first linear layer with number of inputs correspoding to the size of the input feature vector and number of outputs correspoding to the hidden state size
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)

        # second linear layer with number of inputs correspoding to the hidden state size and number of outputs corresponding the number of output classes
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)


    def forward(self, x_in, apply_softmax=False):
        """
        The forward pass of the Classifier

        Args:
            x_in (torch.Tensor): an input data tensor with input shape (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation, which should be false if used with the cross-entropy losses
        Returns:
            resulting tensor, with shape (batch, output_dim)
        """
        # calculate the output of the first linear layer
        y_out = self.fc1(x_in)

        # apply non-linear function to the output of the linear layer
        y_out = F.relu(y_out)

        # calculate the output of the second linear layer
        y_out = self.fc2(y_out)

        # apply softmax function to the calculate output, if needed
        if (apply_softmax):
            y_out = F.softmax(y_out, dim=1)

        return y_out
