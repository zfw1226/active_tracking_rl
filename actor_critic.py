from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils_basic import norm_col_init, normal
import numpy as np
import math


def wrap_action(self, action):
    action = np.squeeze(action)
    out = action * (self.action_high - self.action_low)/2 + (self.action_high + self.action_low)/2.0
    return out


def sample_action(continuous, mu_multi, sigma_multi, device, test=False):
    if continuous:
        mu = torch.clamp(mu_multi, -1.0, 1.0)
        sigma = F.softplus(sigma_multi) + 1e-5
        eps = torch.randn(mu.size())
        pi = np.array([math.pi])
        pi = torch.from_numpy(pi).float()
        eps = Variable(eps).to(device)
        pi = Variable(pi).to(device)
        action = (mu + sigma.sqrt() * eps).data
        act = Variable(action)
        prob = normal(act, mu, sigma, device)
        action = torch.clamp(action, -1.0, 1.0)
        entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)  # 0.5 * (log(2*pi*sigma) + 1
        log_prob = (prob + 1e-6).log()
        action_env = action.cpu().numpy()
    else:  # discrete
        logit = mu_multi
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        if test:
            action = prob.max(1)[1].data
        else:
            action = prob.multinomial(1).data
            log_prob = log_prob.gather(1, Variable(action))
        action_env = action.cpu().numpy()
    return action_env, entropy, log_prob, prob


class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_space, head_name, device):
        super(PolicyNet, self).__init__()
        self.head_name = head_name
        self.device = device
        if 'continuous' in head_name:
            num_outputs = action_space.shape[0]
            self.continuous = True
        else:
            num_outputs = action_space.n
            self.continuous = False


        self.noise = False
        self.actor_linear = nn.Linear(input_dim, num_outputs)
        if self.continuous:
            self.actor_linear2 = nn.Linear(input_dim, num_outputs)

        # init layers
        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.1)
        self.actor_linear.bias.data.fill_(0)
        if self.continuous:
            self.actor_linear2.weight.data = norm_col_init(self.actor_linear2.weight.data, 0.1)
            self.actor_linear2.bias.data.fill_(0)

    def forward(self, x, test=False):
        if self.continuous:
            mu = F.softsign(self.actor_linear(x))
            sigma = self.actor_linear2(x)
        else:
            mu = self.actor_linear(x)
            sigma = torch.ones_like(mu)
            action_env, entropy, log_prob, self.prob = sample_action(self.continuous, mu, sigma, self.device, test)

        return action_env, entropy, log_prob, self.prob


class ValueNet(nn.Module):
    def __init__(self, input_dim, head_name, num=1):
        super(ValueNet, self).__init__()

        self.noise = False
        self.critic_linear = nn.Linear(input_dim, num)
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, x):
        value = self.critic_linear(x)
        return value
