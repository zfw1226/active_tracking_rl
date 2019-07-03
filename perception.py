import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from utils import weights_init


class CNN_simple(nn.Module):
    def __init__(self, obs_shape, stack_frames):
        super(CNN_simple, self).__init__()
        self.conv1 = nn.Conv2d(obs_shape[0], 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        dummy_state = Variable(torch.rand(stack_frames, obs_shape[0], obs_shape[1], obs_shape[2]))
        out = self.forward(dummy_state)
        self.outdim = out.size(-1)
        self.apply(weights_init)
        self.train()

    def forward(self, x):
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = x.view(1, -1)
        return x


class ICML(nn.Module):
    def __init__(self, obs_shape, stack_frames):
        super(ICML, self).__init__()
        self.conv1 = nn.Conv2d(obs_shape[0], 16, 8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)

        dummy_state = Variable(torch.rand(stack_frames, obs_shape[0], obs_shape[1], obs_shape[2]))
        out = self.forward(dummy_state, fc=False)
        cnn_dim = out.size(-1)
        self.fc = nn.Linear(cnn_dim, 256)
        self.outdim = 256
        self.apply(weights_init)
        self.train()

    def forward(self, x, fc=True):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(1, -1)
        if fc:
            x = F.relu(self.fc(x))
        return x


class CNN_maze(nn.Module):
    def __init__(self, obs_shape, stack_frames):
        super(CNN_maze, self).__init__()
        self.conv1 = nn.Conv2d(obs_shape[0], 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)

        dummy_state = Variable(torch.rand(stack_frames, obs_shape[0], obs_shape[1], obs_shape[2]))
        out = self.forward(dummy_state, fc=False)
        cnn_dim = out.size(-1)
        self.fc = nn.Linear(cnn_dim, 256)
        self.outdim = 256
        self.apply(weights_init)
        self.train()

    def forward(self, x, fc=True):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(1, -1)
        if fc:
            x = F.relu(self.fc(x))
        return x


