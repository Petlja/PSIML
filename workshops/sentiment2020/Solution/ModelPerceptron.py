import torch.nn as nn
import torch.nn.functional as F

class SentimentClassifierPerceptron(nn.Module):
    """A simple perceptron based classifier that uses one-hot encoding as an input sequence representation"""

    def __init__(self, num_features, output_dim):
        """
        Args:
            num_features (int): the size of the input feature vector
        """
        # call the base initialization
        super(SentimentClassifierPerceptron, self).__init__()

        # Define the model

        # only one linear layer with number of inputs correspoding to the size of the input feature vector and number of outputs correspoding to binary classification task
        self.fc1 = nn.Linear(in_features=num_features, out_features=output_dim)


    def forward(self, x_in, apply_softmax=False):
        """
        The forward pass of the Classifier

        Args:
            x_in (torch.Tensor): an input data tensor with input shape (batch, num_features)
            apply_softmax (bool): a flag for the softmax activation, which should be false if used with the cross-entropy losses
        Returns:
            resulting tensor, with shape (batch, )
        """
        # calculate the output of the linear layer
        y_out = self.fc1(x_in)

        # apply softmax function to the calculate output, if needed
        if (apply_softmax):
            y_out = F.softmax(y_out, dim=1)

        return y_out
