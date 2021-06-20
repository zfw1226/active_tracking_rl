from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import perception
from actor_critic import PolicyNet, ValueNet


class EncodeTracker(torch.nn.Module):
    def __init__(self, obs_space, lstm_out=128, head_name='cnn_lstm',
                 stack_frames=1, device='cpu'):
        super(EncodeTracker, self).__init__()
        self.head_name = head_name
        self.device = device

        self.encoder = perception.get_cnn_encoder(self.head_name, obs_space, stack_frames)
        feature_dim = self.encoder.outdim
        _, input_dim, self.width, self.height = self.encoder.outshape
        if 'ConvLSTM' in head_name:
            import convLSTM
            self.hiden_dim = input_dim
            self.spatt = convLSTM.ConvLSTMCell(input_dim, self.hiden_dim, 5).to(self.device)
        if 'att' in head_name:
            self.conv_att = torch.nn.Conv2d(input_dim, 16, 1, 1)
            self.conv_feature = torch.nn.Conv2d(input_dim, 16, 1, 1)
            input_dim = 16
            self.pool_att = nn.AdaptiveMaxPool2d((10, 10))
            feature_dim = 10 * 10 * input_dim
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

    def forward(self, inputs):
        x, (hx, cx) = inputs
        batch_size = x.size(0)

        feature_cnn = feature = self.encoder(x, batch_size)
        if 'ConvLSTM' in self.head_name:
            # convlstm attention
            self.hc, self.cc = self.spatt(feature_cnn, self.hc, self.cc)
            feature = self.hc

        if 'att' in self.head_name:
            self.att = torch.nn.functional.hardsigmoid(self.conv_att(feature))
            self.att = self.att.repeat_interleave(int(feature_cnn.shape[1]/self.att.shape[1]), dim=1)
            feature = feature_cnn * self.att
            feature = torch.relu(self.conv_feature(feature))
            feature = self.pool_att(feature)
        if 'pool' in self.head_name:
            feature = self.pool(feature)
        feature = feature.view(batch_size, -1)
        feature_cnn = feature_cnn.view(batch_size, -1)

        if 'layer' in self.head_name:
            feature = self.ln(feature)
            feature_cnn = feature
        if 'lstm' in self.head_name:
            hx, cx = self.lstm(feature, (hx, cx))
            feature = hx  # batch x size
        if 'gru' in self.head_name:
            hx = self.lstm(feature, hx)
            feature = cx = hx
        feature = F.dropout(feature, p=0.5, training=self.training)
        return feature, (hx, cx)

    def reset(self, exampler):
        (batch_size, _, height, width) = exampler.shape
        self.exampler = self.encoder(exampler, batch_size).view(batch_size, -1)

    def init_internal(self, device, batch_size=1):
        # init ConvLSTM internal state
        if 'ConvLSTM' in self.head_name:
            self.hc = torch.zeros(batch_size, self.hiden_dim, self.height, self.width, requires_grad=True).to(device)
            self.cc = torch.zeros(batch_size, self.hiden_dim, self.height, self.width, requires_grad=True).to(device)

class TrackerNet(torch.nn.Module):
    def __init__(self, obs_space, action_space, lstm_out, dim_aux, head_name='cnn_lstm',
                 stack_frames=1, device='cpu', aux='none'):
        super(TrackerNet, self).__init__()
        self.lstm_out = lstm_out
        self.head_name = head_name
        self.aux = aux
        self.encoder_tracker = EncodeTracker(obs_space, lstm_out, head_name, stack_frames, device)
        feature_dim_tracker = self.encoder_tracker.feature_dim
        self.actor_tracker = PolicyNet(feature_dim_tracker, action_space, head_name, device)
        if 'gt' in self.aux:
            feature_dim_critic = dim_aux
        else:
            feature_dim_critic = feature_dim_tracker
        self.critic_tracker = ValueNet(feature_dim_critic, head_name, 1)
        self.reward_aux = nn.Linear(feature_dim_critic, 1)

    def forward(self, obs, test=False, critic=True, aux_info=None): # (states[0], (hx[:1], cx[:1]))
        feature_tracker, (self.hx, self.cx) = self.encoder_tracker((obs, (self.hx, self.cx)))
        action, entropy, log_prob, prob = self.actor_tracker(feature_tracker, test)

        if critic:
            if 'gt' in self.aux:
                critic_input = aux_info
            else:
                critic_input = feature_tracker
            value = self.critic_tracker(critic_input)
            R_pred = self.reward_aux(critic_input)

            return value, action, entropy, log_prob, prob, R_pred
        else:
            return action, entropy, log_prob, prob

    def reset_internal(self, device, batch_size=1):
        self.cx = torch.zeros(batch_size, self.lstm_out, requires_grad=True).to(device)
        self.hx = torch.zeros(batch_size, self.lstm_out, requires_grad=True).to(device)
        self.encoder_tracker.init_internal(device, batch_size)

    def update_internal(self, index=[]):
        for i in index:
            self.cx[i].data = torch.zeros_like(self.cx[i].data)
            self.hx[i].data = torch.zeros_like(self.hx[i].data)
            if 'ConvLSTM' in self.head_name:
                self.encoder_tracker.hc[i].data = torch.zeros_like(self.encoder_tracker.hc[i])
                self.encoder_tracker.cc[i].data = torch.zeros_like(self.encoder_tracker.cc[i])
        # clean gradient
        self.cx = Variable(self.cx.data)
        self.hx = Variable(self.hx.data)
        if 'ConvLSTM' in self.head_name:
            self.encoder_tracker.hc = Variable(self.encoder_tracker.hc.data)
            self.encoder_tracker.cc = Variable(self.encoder_tracker.cc.data)

