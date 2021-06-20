import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from utils_basic import weights_init
import torch.nn.init as init
from convLSTM import ConvLSTMCell
import matplotlib.pyplot as plt

def get_cnn_encoder(head_name, obs_space, stack_frames):
    if 'cnn' in head_name:
        encoder = CNN_simple(obs_space, stack_frames)
    elif 'icml' in head_name:
        encoder = ICML(obs_space, stack_frames)
    elif 'tiny' in head_name:
        encoder = Tiny(obs_space, stack_frames)

    return encoder

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
        self.outshape = out.shape
        out = out.view(stack_frames, -1)
        cnn_dim = out.size(-1)
        self.outdim = cnn_dim
        self.apply(weights_init)
        self.train()

    def forward(self, x, batch_size=1, fc=False):
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        return x

class Tiny(nn.Module):
    def __init__(self, obs_shape, stack_frames, norm='none'):
        super(Tiny, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 5, 2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1, groups=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1),
            nn.LeakyReLU(inplace=True),
        )

        dummy_state = Variable(torch.rand(stack_frames, obs_shape[0], obs_shape[1], obs_shape[2]))
        out = self.forward(dummy_state, fc=False)
        self.outshape = out.shape
        out = out.view(stack_frames, -1)
        cnn_dim = out.size(-1)
        self.outdim = cnn_dim
        self.apply(weights_init)
        self.train()

    def forward(self, x, batch_size=1, fc=False):
        x = self.features(x)
        if fc:
            x = x.view(batch_size, -1)
            x = F.relu(self.fc(x))
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

    def forward(self, x, batch_size=1, fc=True):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)
        if fc:
            x = F.relu(self.fc(x))
        return x

class BiRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, head_name):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if 'lstm' in head_name:
            self.lstm = True
        else:
            self.lstm = False
        if self.lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.feature_dim = hidden_size * 2
        self.device = device

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # 2 for bidirection

        # Forward propagate LSTM
        if self.lstm:
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
            out, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        else:
            out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        return out

class ConvLSTM(nn.Module):
    def __init__(self, feature_shape, device):
        super(ConvLSTM, self).__init__()

        self.input_channels = [feature_shape[1]]  #  + self.hidden_channels
        self.num_layers = len(self.hidden_channels)
        self.kernel_size = 3
        self._all_layers = []
        self.device = device
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size).to(device)
            setattr(self, name, cell)
            self._all_layers.append(cell)
        self.obs_shape = feature_shape
        self.init_hidden()
        dummy_state = Variable(torch.rand(feature_shape).to(device))
        out = self.forward(dummy_state)
        out = out.view(1, -1)
        cnn_dim = out.size(-1)
        self.outdim = cnn_dim
        self.train()

    def forward(self, x, batch_size=1):
        for i in range(self.num_layers):
            # all cells are initialized in the first step
            name = 'cell{}'.format(i)
            # do forward
            (h, c) = self.internal_state[i]
            x, new_c = getattr(self, name)(x, h, c)
            self.internal_state[i] = (x, new_c)
            x = self.maxp(x)
        x = x.view(batch_size, -1)
        return x

    def init_hidden(self, bsize=1):
        hxs = []
        chx = []
        x = torch.rand(bsize, self.obs_shape[1], self.obs_shape[2], self.obs_shape[3]).to(self.device)
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            _, _, height, width = self.obs_shape
            (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                     shape=(height, width), device=self.device)
            hxs.append((h, c))
            x, new_c = getattr(self, name)(x, h, c)

    def reset_one(self, index):
        for l_i in range(len(self.internal_state)):
            (h, c) = self.internal_state[l_i]
            h[index].data = torch.zeros_like(h[index].data)
            c[index].data = torch.zeros_like(c[index].data)
            self.internal_state[l_i] = (h, c)