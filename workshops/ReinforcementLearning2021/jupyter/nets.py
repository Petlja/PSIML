import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, s_size, a_size, h=16, num_hidden=1):
        self.state_size = s_size
        self.action_size = a_size

        modules = [
            nn.Linear(self.state_size, h, dtype=torch.float32),
            nn.ReLU()
        ]
        for _ in range(num_hidden):
            modules.append(nn.Linear(h, h, dtype=torch.float32))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(h, self.action_size, dtype=torch.float32))

        super(DQN, self).__init__()
        self.neuralnet = nn.Sequential(*modules)

    def forward(self, x):
        x = x.type(torch.float32)
        return self.neuralnet(x)


class ContinuousQNet(nn.Module):
    def __init__(self, s_size, a_size, h=16, num_hidden=1):
        self.state_size = s_size
        self.action_size = a_size

        # Define network input and output dimensions
        # Hint: We cannot do max Q(s,a) over the network output anymore
        # so action must be handled differently!
        input_dim = s_size + a_size
        output_dim = 1

        modules = [
            nn.Linear(input_dim, h, dtype=torch.float32),
            nn.ReLU()
        ]
        for _ in range(num_hidden):
            modules.append(nn.Linear(h, h, dtype=torch.float32))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(h, output_dim, dtype=torch.float32))

        super(ContinuousQNet, self).__init__()
        self.neuralnet = nn.Sequential(*modules)

    def forward(self, x):
        x = x.type(torch.float32)
        return self.neuralnet(x)


class Actor(nn.Module):
    def __init__(self, s_size, a_size, a_limit, h=16, num_hidden=1):
        self.state_size = s_size
        self.action_size = a_size
        self.action_limit = a_limit

        modules = [
            nn.Linear(self.state_size, h, dtype=torch.float32),
            nn.ReLU()
        ]
        for i in range(num_hidden):
            modules.append(nn.Linear(h, h, dtype=torch.float32))
            if i != num_hidden - 1:
                modules.append(nn.ReLU())
            else:
                modules.append(nn.Tanh())
        modules.append(nn.Linear(h, a_size, dtype=torch.float32))

        super(Actor, self).__init__()
        self.neuralnet = nn.Sequential(*modules)

    def forward(self, x):
        x = x.type(torch.float32)
        return self.neuralnet(x).clamp_(-self.action_limit, self.action_limit)
