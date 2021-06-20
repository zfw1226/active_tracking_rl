from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import perception as perception
from model import PolicyNet, ValueNet

class EncodeTarget(torch.nn.Module):
    def __init__(self, obs_space, lstm_out=128, head_name='(opp)-(self)-nips-start-layer-lstm/gru',
                 stack_frames=1, f_aux_dim=-1):
        super(EncodeTarget, self).__init__()
        self.head_name = head_name
        self.f_aux_dim = f_aux_dim
        if 'novision' in self.head_name:
            feature_dim = 0
        else:
            self.encoder = perception.get_cnn_encoder(self.head_name, obs_space, stack_frames)

            feature_dim = self.encoder.outdim
            if 'opp' in self.head_name and 'self' in self.head_name:
                feature_dim += feature_dim

        if self.f_aux_dim > 0:
                feature_dim = feature_dim + self.f_aux_dim

        if 'layer' in head_name:
            self.ln = nn.LayerNorm(feature_dim)

        if 'lstm' in head_name:
            self.lstm = nn.LSTMCell(feature_dim, lstm_out)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)
            feature_dim = lstm_out
        if 'gru' in head_name:
            self.lstm = nn.GRUCell(feature_dim, lstm_out)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)
            feature_dim = lstm_out

        self.feature_dim = feature_dim
        self.train()

    def forward(self, inputs, f_aux=None):
        x, (hx, cx) = inputs
        batch_size = x.size(0)
        if 'novision' not in self.head_name:
            feature = self.encoder(x, batch_size)
            feature = feature.view(batch_size, -1)
            if 'opp' in self.head_name and 'self' in self.head_name: # x = (self_0, self_1, opp_0, opp_1)
                feature_self, feature_opp = torch.split(feature, int(batch_size/2), dim=0)
                feature = torch.cat((feature_self, feature_opp), 1)

        if self.f_aux_dim > 0:
            if 'novision' in self.head_name:
                feature = f_aux
            else:
                feature = torch.cat((feature, f_aux), 1)
        if 'layer' in self.head_name:
            feature = self.ln(feature)
        if 'lstm' in self.head_name:
            hx, cx = self.lstm(feature, (hx, cx))
            feature = hx
        if 'gru' in self.head_name:
            hx = self.lstm(feature, hx)
            feature = hx

        return feature, (hx, cx)


class TargetNet(torch.nn.Module):
    def __init__(self, obs_space, action_space, lstm_out=128, dim_aux=128,  head_name='cnn_lstm',
                 stack_frames=1, device='cpu', aux='none'):
        super(TargetNet, self).__init__()
        self.head_name = head_name
        self.lstm_out = lstm_out
        self.aux = aux
        self.encoder = EncodeTarget(obs_space, self.lstm_out, head_name, stack_frames, dim_aux)
        feature_dim = self.encoder.feature_dim
        self.critic = ValueNet(feature_dim, head_name, 1)
        self.actor = PolicyNet(feature_dim, action_space, head_name, device)
        self.reward_aux = nn.Linear(feature_dim, 1)

    def forward(self, obs_target, info_aux, obs_tracker, test=False):  # input pos(feature), action_tracker, obs_tracker, obs
        if 'opp' in self.head_name and 'self' in self.head_name:
            x = torch.cat([obs_target, obs_tracker], 0)
        elif 'opp' in self.head_name:
            x = obs_tracker
        else:
            x = obs_target
        feature, (self.hx, self.cx) = self.encoder((x, (self.hx, self.cx)), info_aux)
        action, entropy, log_prob, prob = self.actor(feature, test)
        value = self.critic(feature)
        R_pred = self.reward_aux(feature)
        return value, action, entropy, log_prob, prob, R_pred

    def reset_internal(self, device, batch_size=1):
        self.cx = torch.zeros(batch_size, self.lstm_out, requires_grad=True).to(device)
        self.hx = torch.zeros(batch_size, self.lstm_out, requires_grad=True).to(device)

    def update_internal(self, index=[]):
        for i in index:
            self.cx[i].data = torch.zeros_like(self.cx[i].data)
            self.hx[i].data = torch.zeros_like(self.hx[i].data)

        # clean gradient
        self.cx = Variable(self.cx.data)
        self.hx = Variable(self.hx.data)

