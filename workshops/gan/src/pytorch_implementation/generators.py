import torch
import torch.nn as nn

class FCGenerator(nn.Module):
    def __init__(self, image_size, channels, z_size):
        """
        Network which takes a batch of random vectors and creates images out of them.

        :param img_size: width and height of the image
        :param channels: number of channels
        """
        super(FCGenerator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.fc1 = nn.Linear(z_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, image_size*image_size*channels)
        self.relu = nn.ReLU()

    def forward(self, z_batch):
        x = self.fc1(z_batch)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)

        image = x.reshape(-1, self.channels, self.image_size, self.image_size)
        return image

