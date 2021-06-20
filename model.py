from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from perception import BiRNN
import numpy as np
from actor_critic import PolicyNet, ValueNet
from tracker_net import TrackerNet
from target_net import TargetNet


def build_model(obs_space, action_space, args, device):
    name = args.model
    if 'simple' in name:
        model = A3C_MULTI_POS(obs_space, action_space, args, device)
    elif 'sep' in name:
        model = A3C_MULTI_SEP(obs_space, action_space, args, device)
    else:
        model = A3C_MULTI_TRACK(obs_space, action_space, args, device)

    model.train()
    return model

class A3C_MULTI_POS(torch.nn.Module):
    def __init__(self, obs_space, action_spaces, args, device=None):
        super(A3C_MULTI_POS, self).__init__()
        self.num_agents = len(obs_space)
        self.lstm_out = args.rnn_teacher
        fuse_out = args.fuse_out
        head_name = args.model
        if 'reward' in args.aux:
            self.sub_task = True
        else:
            self.sub_task = False

        self.head_name = head_name
        if 'continuous' in head_name:
            self.continuous = True
            self.action_dim = action_spaces[0].shape[0]
        else:
            self.continuous = False
            self.action_dim = action_spaces[0].n

        pos_dim = args.pos
        self.encoder_pos = BiRNN(pos_dim, int(fuse_out / 2), 1, device,
                                 'gru')  # input_size, hiden_size, layers, device
        self.encoder_pos_tracker = BiRNN(pos_dim, int(fuse_out / 2), 1, device,
                                         'gru')  # input_size, hiden_size, layers, device

        feature_dim_pos = self.encoder_pos.feature_dim

        # config aux info
        if 'act' in self.head_name and 'pos' in self.head_name:
            fuse_dim = feature_dim_pos + self.action_dim
        elif 'act' in self.head_name:
            fuse_dim = self.action_dim
        elif 'pos' in self.head_name:
            fuse_dim = feature_dim_pos
        else:
            fuse_dim = -1

        self.rnn = False
        if 'lstm' in head_name:
            self.rnn = True
            # tracker
            self.lstm_tracker = nn.LSTMCell(feature_dim_pos, self.lstm_out)
            self.lstm_tracker.bias_ih.data.fill_(0)
            self.lstm_tracker.bias_hh.data.fill_(0)
            feature_dim_pos = self.lstm_out
            # target / distractor
            self.lstm_target = nn.LSTMCell(fuse_dim, self.lstm_out)
            self.lstm_target.bias_ih.data.fill_(0)
            self.lstm_target.bias_hh.data.fill_(0)
            fuse_dim = self.lstm_out

        # create actor
        self.critic_tracker = ValueNet(feature_dim_pos, head_name, 1)
        self.critic_target = ValueNet(fuse_dim, head_name, 1)
        self.critic_distractor = ValueNet(fuse_dim, head_name, 1)

        self.actor_tracker = PolicyNet(feature_dim_pos, action_spaces[0], head_name, device)
        self.actor_target = PolicyNet(fuse_dim, action_spaces[1], head_name, device)
        self.actor_distractor = PolicyNet(fuse_dim, action_spaces[1], head_name, device)

        if self.sub_task:
            self.reward_aux = nn.Linear(feature_dim_pos, 1)
            self.reward_aux_god = nn.Linear(fuse_dim, 1)

        # self.apply(weights_init)
        self.train()
        self.device = device

    def forward(self, inputs, test=False):
        R_pred = torch.zeros(1).to(self.device)
        in_pred = torch.zeros(1).to(self.device)
        states, pos_obs = inputs
        hx = self.hxs
        cx = self.cxs
        entropies = []
        log_probs = []
        probs = []
        actions = []
        values = []
        if 'pos' in self.head_name:
            feature_pos = self.encoder_pos(Variable(pos_obs, requires_grad=True))[:, -1, :]
            feature_pos_tracker = self.encoder_pos_tracker(Variable(pos_obs[:1], requires_grad=True))[:, 0, :]
        num_agents = states.shape[0]
        for i in range(num_agents):
            if i == 0:
                if self.rnn:
                    (hx_out, cx_out) = self.lstm_tracker(feature_pos_tracker, (hx[:1], cx[:1]))
                    feature_pos_tracker = hx_out
                # input to action
                action, entropy, log_prob, prob = self.actor_tracker(feature_pos_tracker, test)
                action = np.squeeze(action)

                # action to others
                action_onehot = torch.zeros(1, self.action_dim)
                action_onehot[0][action] = 1
                if num_agents > 1:
                    action_onehot = action_onehot.repeat(num_agents-1, 1).to(self.device)

                # learning with aux task
                if self.sub_task:
                    R_pred = self.reward_aux(feature_pos_tracker)

                value = self.critic_tracker(feature_pos_tracker)
            elif i == 1:
                if 'act' in self.head_name and 'pos' in self.head_name:
                    feature_fuse = torch.cat((feature_pos[1:num_agents], action_onehot), 1)
                elif 'act' in self.head_name:
                    feature_fuse = action_onehot
                elif 'pos' in self.head_name:
                    feature_fuse = feature_pos[1:num_agents]

                if self.rnn:
                    (hx_out_tmp, cx_out_tmp) = self.lstm_target(feature_fuse, (hx[1:], cx[1:]))
                    feature_fuse = hx_out_tmp
                    hx_out = torch.cat((hx_out, hx_out_tmp))
                    cx_out = torch.cat((cx_out, cx_out_tmp))

                feature_target = feature_fuse[i - 1].unsqueeze(0)
                if self.sub_task:
                    R_p = self.reward_aux_god(feature_target)
                    R_pred = torch.cat([R_pred, R_p])
                action, entropy, log_prob, prob = self.actor_target(feature_target, test)
                action = np.squeeze(action)
                value = self.critic_target(feature_target)
            else:
                feature_distractor = feature_fuse[i - 1].unsqueeze(0)
                action, entropy, log_prob, prob = self.actor_distractor(feature_distractor, test)
                action = np.squeeze(action)
                value = self.critic_distractor(feature_distractor)

            probs.append(prob)
            log_probs.append(log_prob)
            entropies.append(entropy)
            actions.append(action)
            values.append(value)

        self.probs = torch.cat(probs)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        values = torch.cat(values)
        if 'continuous' in self.head_name:
            entropies = entropies.sum(-1)
            entropies = entropies.unsqueeze(1)
            log_probs = log_probs.sum(-1)
            log_probs = log_probs.unsqueeze(1)

        self.hxs, self.cxs = hx_out, cx_out
        return values, actions, entropies, log_probs, (R_pred, in_pred)

    def reset_hiden(self, num_agents):
        self.cxs = torch.zeros(num_agents, self.lstm_out, requires_grad=True).to(self.device)
        self.hxs = torch.zeros(num_agents, self.lstm_out, requires_grad=True).to(self.device)

    def update_hiden(self):
        self.cxs = Variable(self.cxs.data)
        self.hxs = Variable(self.hxs.data)

    def generate_pattern(self, num_agents):
        self.pz = torch.randn(num_agents-1, 128)
        # self.pz = self.pz.repeat(num_agents-1, 1)


class A3C_MULTI_TRACK(torch.nn.Module):
    def __init__(self, obs_space, action_spaces, args, device=None):
        super(A3C_MULTI_TRACK, self).__init__()
        self.num_agents = len(obs_space)
        self.lstm_out = args.rnn_teacher
        fuse_out = args.fuse_out
        head_name = args.model
        stack_frames = args.stack_frames

        obs_shapes = [obs_space[i].shape for i in range(self.num_agents)]
        if self.num_agents == 1:
            obs_shapes = obs_shapes + obs_shapes + obs_shapes
            action_spaces = action_spaces + action_spaces + action_spaces

        self.head_name = head_name

        if 'continuous' in head_name:
            self.continuous = True
            self.action_dim = action_spaces[0].shape[0]
        else:
            self.continuous = False
            self.action_dim = action_spaces[0].n

        # config aux info
        pos_dim = args.pos
        if 'pos' in self.head_name:
            self.encoder_pos = BiRNN(pos_dim, int(fuse_out / 2), 1, device, 'gru')  # input_size, hiden_size, layers, device
            self.encoder_pos_tracker = BiRNN(pos_dim, int(fuse_out / 2), 1, device, 'gru')  # input_size, hiden_size, layers, device
        feature_dim_pos = self.encoder_pos.feature_dim

        self.tracker = TrackerNet(obs_shapes[0], action_spaces[0], args.rnn_out, feature_dim_pos, args.tracker_net, stack_frames, device, args.aux)

        # config aux info
        if 'act' in self.head_name and 'pos' in self.head_name:
            fuse_dim = feature_dim_pos + self.action_dim
        elif 'act' in self.head_name:
            fuse_dim = self.action_dim
        elif 'pos' in self.head_name:
            fuse_dim = feature_dim_pos
        else:
            fuse_dim = -1

        self.rnn = False
        if 'lstm' in head_name:
            # target / distractor
            self.lstm_target = nn.LSTMCell(fuse_dim, self.lstm_out)
            self.lstm_target.bias_ih.data.fill_(0)
            self.lstm_target.bias_hh.data.fill_(0)
            fuse_dim = self.lstm_out
            self.rnn = True

        # create actor
        self.critic_target = ValueNet(fuse_dim, head_name, 1)
        self.critic_distractor = ValueNet(fuse_dim, head_name, 1)

        self.actor_target = PolicyNet(fuse_dim, action_spaces[1], head_name, device)
        self.actor_distractor = PolicyNet(fuse_dim, action_spaces[1], head_name, device)

        if 'reward' in args.aux:
            self.sub_task = True
            self.reward_aux_god = nn.Linear(fuse_dim, 1)
        else:
            self.sub_task = False

        # self.apply(weights_init)
        self.train()
        self.device = device

    def forward(self, inputs, test=False):
        R_pred = torch.zeros(1).to(self.device)
        in_pred = torch.zeros(1).to(self.device)
        states, pos_obs = inputs
        hx = self.hxs
        cx = self.cxs
        entropies = []
        log_probs = []
        probs = []
        actions = []
        values = []
        if 'pos' in self.head_name:
            if 'share' in self.head_name:
                feature_pos = self.encoder_pos(Variable(pos_obs[:1], requires_grad=True))[0]
                feature_pos_tracker = self.encoder_pos(Variable(pos_obs[:1], requires_grad=True))[:, 0, :]
            else:
                feature_pos = self.encoder_pos(Variable(pos_obs, requires_grad=True))[:, -1, :]
                feature_pos_tracker = self.encoder_pos_tracker(Variable(pos_obs[:1], requires_grad=True))[:, 0, :]
        num_agents = states.shape[0]
        # states = states
        for i in range(num_agents):
            if i == 0:
                # input to action
                value, action, entropy, log_prob, prob, R_pred = self.tracker(Variable(states[0], requires_grad=True), test, True, feature_pos_tracker)

                # action to others
                action_onehot = torch.zeros(1, self.action_dim)
                action_onehot[0][action] = 1
                if num_agents > 1:
                    action_onehot = action_onehot.repeat(num_agents-1, 1).to(self.device)
                action = np.squeeze(action)
            elif i == 1:
                if 'act' in self.head_name and 'pos' in self.head_name:
                    feature_fuse = torch.cat((feature_pos[1:num_agents], action_onehot), 1)
                elif 'act' in self.head_name:
                    feature_fuse = action_onehot
                elif 'pos' in self.head_name:
                    feature_fuse = feature_pos[1:num_agents]

                if self.rnn:
                    (hx_out, cx_out) = self.lstm_target(feature_fuse, (hx, cx))
                    self.hxs = hx_out
                    self.cxs = cx_out
                    feature_fuse = hx_out

                feature_target = feature_fuse[i - 1].unsqueeze(0)
                if self.sub_task:
                    R_p = self.reward_aux_god(feature_target)
                    R_pred = torch.cat([R_pred, R_p])
                action, entropy, log_prob, prob = self.actor_target(feature_target, test)
                action = np.squeeze(action)
                value = self.critic_target(feature_target)
            else:
                feature_distractor = feature_fuse[i - 1].unsqueeze(0)
                action, entropy, log_prob, prob = self.actor_distractor(feature_distractor, test)
                action = np.squeeze(action)
                value = self.critic_distractor(feature_distractor)

            probs.append(prob)
            log_probs.append(log_prob)
            entropies.append(entropy)
            actions.append(action)
            values.append(value)

        self.probs = torch.cat(probs)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        values = torch.cat(values)
        if 'continuous' in self.head_name:
            entropies = entropies.sum(-1)
            entropies = entropies.unsqueeze(1)
            log_probs = log_probs.sum(-1)
            log_probs = log_probs.unsqueeze(1)
        return values, actions, entropies, log_probs, (R_pred, in_pred)

    def reset_hiden(self, num_agents, device=None):
        if device is None:
            device = self.device
        self.tracker.reset_internal(device)
        self.cxs = torch.zeros(num_agents-1, self.lstm_out, requires_grad=True).to(self.device)
        self.hxs = torch.zeros(num_agents-1, self.lstm_out, requires_grad=True).to(self.device)

    def update_hiden(self):
        self.tracker.update_internal()
        self.cxs = Variable(self.cxs.data)
        self.hxs = Variable(self.hxs.data)


class A3C_MULTI_SEP(torch.nn.Module):
    def __init__(self, obs_space, action_spaces, args, device=None):
        super(A3C_MULTI_SEP, self).__init__()
        self.num_agents = len(obs_space)
        self.lstm_out = args.rnn_out
        self.lstm_teacher = args.rnn_teacher
        fuse_out = args.fuse_out
        head_name = args.model
        stack_frames = args.stack_frames

        obs_shapes = [obs_space[i].shape for i in range(self.num_agents)]
        if self.num_agents == 1:
            obs_shapes = obs_shapes + obs_shapes + obs_shapes
            action_spaces = action_spaces + action_spaces + action_spaces

        self.head_name = head_name

        if 'continuous' in head_name:
            self.continuous = True
            self.action_dim = action_spaces[0].shape[0]
        else:
            self.continuous = False
            self.action_dim = action_spaces[0].n

        # config aux info
        pos_dim = args.pos
        if 'pos' in self.head_name:
            self.encoder_pos = BiRNN(pos_dim, int(fuse_out / 2), 1, device, 'gru')  # input_size, hiden_size, layers, device
            self.encoder_pos_tracker = BiRNN(pos_dim, int(fuse_out / 2), 1, device, 'gru')  # input_size, hiden_size, layers, device
        feature_dim_pos = self.encoder_pos.feature_dim
        if 'act' in self.head_name:
            dim_aux = feature_dim_pos + self.action_dim
        else:
            dim_aux = feature_dim_pos

        self.tracker = TrackerNet(obs_shapes[0], action_spaces[0], self.lstm_out, feature_dim_pos, args.tracker_net, stack_frames, device, args.aux)

        self.target = TargetNet(obs_shapes[1], action_spaces[1], self.lstm_out, dim_aux, head_name, stack_frames, device)

        self.distractor = TargetNet(obs_shapes[2], action_spaces[2], self.lstm_out, dim_aux, head_name, stack_frames, device)

        # self.apply(weights_init)
        self.train()
        self.device = device

    def forward(self, inputs, test=False):
        R_in = torch.zeros(1).to(self.device)
        states, pos_obs = inputs
        if 'pos' in self.head_name:
            if 'share' in self.head_name:
                feature_pos = self.encoder_pos(Variable(pos_obs[:1], requires_grad=True))[0]
                feature_pos_tracker = self.encoder_pos(Variable(pos_obs[:1], requires_grad=True))[:, 0, :]
            else:
                feature_pos = self.encoder_pos(Variable(pos_obs, requires_grad=True))[:, -1, :]
                feature_pos_tracker = self.encoder_pos_tracker(Variable(pos_obs[:1], requires_grad=True))[:, 0, :]

        num_agents = states.shape[0]
        # states = states
        for i in range(min(3, num_agents)):
            if i == 0:
                # input to action
                value, action, entropy, log_prob, prob, R_pred = self.tracker(Variable(states[0], requires_grad=True), test, True, feature_pos_tracker)

                # action to others
                action_onehot = torch.zeros(1, self.action_dim)
                action_onehot[0][action] = 1
                if num_agents > 1:
                    action_onehot = action_onehot.repeat(num_agents-1, 1).to(self.device)

            elif i == 1:
                # prepare info_aux, including used for distractor
                if 'act' in self.head_name:
                    info_aux = torch.cat((feature_pos[1:num_agents], action_onehot), 1)
                else:
                    info_aux = feature_pos[1:num_agents]
                value, action, entropy, log_prob, prob, R_pred = self.target(states[1], info_aux[:1], states[0], test)

            elif i == 2:
                # need to batch
                if num_agents-2 > 1:
                    obs = states[2:].squeeze(1)
                    obs_tracker = states[0].expand_as(obs)
                else:
                    obs_tracker = states[0]
                    obs = states[i]
                value, action, entropy, log_prob, prob, R_pred = self.distractor(obs, info_aux[i-1:], obs_tracker, test)

            if i == 0:
                entropies = entropy
                log_probs = log_prob
                probs = prob
                actions = action
                values = value
                R_preds = R_pred
            else:
                probs = torch.cat([probs, prob])
                log_probs = torch.cat([log_probs, log_prob])
                entropies = torch.cat([entropies, entropy])
                values = torch.cat([values, value])
                actions = np.concatenate([actions, action])
                R_preds = torch.cat([R_preds, R_pred])
        if 'continuous' in self.head_name:
            entropies = entropies.sum(-1)
            entropies = entropies.unsqueeze(1)
            log_probs = log_probs.sum(-1)
            log_probs = log_probs.unsqueeze(1)
        if len(actions.shape) == 2:
            actions = np.squeeze(actions, 1)
        return values, actions, entropies, log_probs, (R_preds, R_in)

    def reset_hiden(self, num_agents, device=None):
        if device is None:
            device = self.device
        self.tracker.reset_internal(device)
        self.target.reset_internal(device)
        self.distractor.reset_internal(device, max(num_agents-2, 0))

    def update_hiden(self):
        self.tracker.update_internal()
        self.target.update_internal()
        self.distractor.update_internal()