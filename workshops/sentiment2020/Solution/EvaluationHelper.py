import torch.nn.functional as F

def compute_accuracy(output, labels, apply_softmax=True):
    """
    Computes accuracy of the predicted outputs compared to the labels

    Args:
        outputs (torch.Tensor): output predictions of the model
        labels (torch.Tensor): labels for corresponding data points from the dataset
        apply_softmax (bool): a flag if the softmax function should be applied on the outputs before the accuracy calculation
    """
    # firstly apply softmax to the outputs, if needed
    if (apply_softmax):
        output = F.softmax(output, dim=1)

    # find the best class as the one represented with highest probability
    _, indices = output.max(dim=1)

    # calculate number of correctly predicted instances
    correct = (indices == labels).float().sum()

    return correct / len(labels)