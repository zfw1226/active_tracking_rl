from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import L1Loss
from utils import ensure_shared_grads


class Agent(object):
    def __init__(self, model, env, args, state, device):
        self.model = model
        self.env = env
        self.num_agents = len(env.action_space)
        if 'continuous' in args.network:
            if type(env.action_space) == list:
                self.action_high = [env.action_space[i].high for i in range(self.num_agents)]
                self.action_low = [env.action_space[i].low for i in range(self.num_agents)]
            else:
                self.action_high = [env.action_space.high, env.action_space.high]
                self.action_low = [env.action_space.low, env.action_space.low]

        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.preds = []
        self.done = True
        self.info = None
        self.reward = 0
        self.device = device
        self.rnn_out = args.rnn_out
        self.num_steps = 0
        self.n_steps = 0
        self.state = state
        self.hxs = torch.zeros(self.num_agents, self.rnn_out).to(device)
        self.cxs = torch.zeros(self.num_agents, self.rnn_out).to(device)

    def wrap_action(self, action, high, low):
        action = np.squeeze(action)
        out = action * (high - low)/2.0 + (high + low)/2.0
        return out

    def action_train(self):
        self.n_steps += 1
        value_multi, action_env_multi, entropy, log_prob, (self.hxs, self.cxs), R_pred = self.model(
            (Variable(self.state, requires_grad=True), (self.hxs, self.cxs)))

        if 'continuous' in self.args.network:
            action_env_multi = [self.wrap_action(action_env_multi[i], self.action_high[i], self.action_low[i])
                                for i in range(self.num_agents)]

        # model return action_env_multi, entropy, log_prob
        state_multi, reward_multi, self.done, self.info = self.env.step(action_env_multi)

        # add to buffer
        self.reward_org = reward_multi.copy()
        self.reward = torch.tensor(reward_multi).float().to(self.device)
        self.state = torch.from_numpy(state_multi).float().to(self.device)
        self.eps_len += 1
        self.values.append(value_multi)

        self.entropies.append(entropy)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward.unsqueeze(1))
        self.preds.append(R_pred)
        return self

    def action_test(self):
        with torch.no_grad():
            value_multi, action_env_multi, entropy, log_prob, (self.hxs, self.cxs), R_pred = self.model(
                (Variable(self.state), (self.hxs, self.cxs)), True)

        if 'continuous' in self.args.network:
            action_env_multi = [self.wrap_action(action_env_multi[i], self.action_high[i], self.action_low[i])
                                for i in range(self.num_agents)]

        state_multi, self.reward, self.done, self.info = self.env.step(action_env_multi)

        self.state = torch.from_numpy(state_multi).float().to(self.device)
        self.eps_len += 1
        return self

    def reset(self):
        self.state = torch.from_numpy(self.env.reset()).float().to(self.device)
        self.num_agents = self.state.shape[0]
        self.eps_len = 0
        self.reset_rnn_hiden()

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.preds = []
        return self

    def reset_rnn_hiden(self):
        self.cxs = torch.zeros(self.num_agents, self.rnn_out).to(self.device)
        self.hxs = torch.zeros(self.num_agents, self.rnn_out).to(self.device)
        self.cxs = Variable(self.cxs)
        self.hxs = Variable(self.hxs)

    def update_rnn_hiden(self):
        self.cxs = Variable(self.cxs.data)
        self.hxs = Variable(self.hxs.data)

    def optimize(self, params, optimizer, shared_model, training_mode, gpu_id):
        R = torch.zeros(self.num_agents, 1).to(self.device)
        if not self.done:
            # predict value
            state = self.state
            value_multi, _, _, _, _, _ = self.model(
                (Variable(state, requires_grad=True), (self.hxs, self.cxs)))
            for i in range(self.num_agents):
                R[i][0] = value_multi[i].data
        self.values.append(Variable(R).to(self.device))
        policy_loss = torch.zeros(self.num_agents, 1).to(self.device)
        value_loss = torch.zeros(self.num_agents, 1).to(self.device)
        pred_loss = torch.zeros(1, 1).to(self.device)
        entropies = torch.zeros(self.num_agents, 1).to(self.device)
        w_entropies = float(self.args.entropy)*torch.ones(self.num_agents, 1).to(self.device)
        w_entropies[1][0] = float(self.w_entropy_coach)
        R = Variable(R, requires_grad=True).to(self.device)
        gae = torch.zeros(1, 1).to(self.device)
        l1_loss = L1Loss()
        for i in reversed(range(len(self.rewards))):
            if 'reward' in self.args.aux:
                pred_loss = pred_loss + l1_loss(self.preds[i][0], self.rewards[i][0])
            R = self.args.gamma * R + self.rewards[i]
            advantage = R - self.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            # Generalized Advantage Estimataion
            delta_t = self.rewards[i] + self.args.gamma * self.values[i + 1].data - self.values[i].data
            gae = gae * self.args.gamma * self.args.tau + delta_t
            policy_loss = policy_loss - \
                (self.log_probs[i] * Variable(gae)) - \
                (w_entropies * self.entropies[i])
            entropies += self.entropies[i]

        self.model.zero_grad()
        loss_tracker = (policy_loss[0] + 0.5 * value_loss[0]).mean()
        loss_target = (policy_loss[1] + 0.5 * value_loss[1]).mean()

        if training_mode == 0:  # train tracker
            loss = loss_tracker
        elif training_mode == 1:  # train target
            loss = loss_target
        else:
            loss = loss_tracker + loss_target
        if 'reward' in self.args.aux and training_mode != 0:
            loss += pred_loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(params, 50)
        ensure_shared_grads(self.model, shared_model, gpu=gpu_id >= 0)

        optimizer.step()
        self.clear_actions()
        return policy_loss, value_loss, entropies, pred_loss