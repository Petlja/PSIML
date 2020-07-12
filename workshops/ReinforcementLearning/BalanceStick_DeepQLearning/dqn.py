import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        h = 24 #hidden dimension

        super(DQN, self).__init__()
        self.neuralnet = nn.Sequential(
            nn.Linear(self.state_size, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, self.action_size))

    def forward(self, x):
        return self.neuralnet(x)