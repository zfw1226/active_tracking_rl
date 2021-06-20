from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import L1Loss
from utils_basic import ensure_shared_grads, load_weight
import os, random
import cv2
from model import A3C_MULTI_POS
import matplotlib.pyplot as plt

class Agent(object):
    def __init__(self, model, env, args, state, device):
        self.model = model
        self.env = env
        self.num_agents = len(env.action_space)
        self.num_agents = len(env.observation_space)
        if 'continuous' in args.model:
            self.continuous = True
            self.action_high = [env.action_space[i].high for i in range(self.num_agents)]
            self.action_low = [env.action_space[i].low for i in range(self.num_agents)]
            self.dim_action = env.action_space[0].shape[0]
        else:
            self.dim_action = 1
            self.continuous = False
        if 'teacher' in args.aux or 'attack' in args.aux:
            self.teacher = A3C_MULTI_POS(env.observation_space, env.action_space, args, device)
            self.teacher, _ = load_weight(self.teacher, args.load_teacher)
            self.teacher.to(device)
            self.teacher.eval()
        self.eps_len = 0
        self.eps_num = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.teacher_prob = []
        self.rewards = []
        self.entropies = []
        self.R_preds = []
        self.in_preds = []
        self.in_labels = []
        self.perfects = []
        self.obs_tracker = []
        self.obs = []
        self.obs_tracker_eps = []
        self.rewards_eps = []
        self.action_prob_eps = []
        self.pos_obs_eps = []
        self.in_labels_eps = []
        self.done = True
        self.info = None
        self.reward = 0
        self.device = device
        self.lstm_out = args.rnn_out
        self.num_steps = 0
        self.n_steps = 0
        self.state = state
        self.rank = 0
        self.norm_r = [reward_normalizer(), reward_normalizer(), reward_normalizer()]

        if 'rule' in self.args.tracker_net:
            self.rule_actor = rule_ctl(self.env.resolution)
    def wrap_action(self, action, high, low):
        action = np.squeeze(action)
        out = action * (high - low)/2.0 + (high + low)/2.0
        return out

    def action_train(self):
        self.n_steps += 1

        value_multi, action_env_multi, entropy, log_prob, (R_pred, in_pred) = self.model(
            (Variable(self.state, requires_grad=True), self.pos_obs[:, :, :self.args.pos]))
        if 'rule' in self.args.tracker_net:
            action_env_multi[0] = self.rule_actor.act(self.env.bbox_pred)

        if 'attack' in self.args.aux and self.num_agents > 1:
            value_teacher, action_env_teacher, _, _, _ = self.teacher(
                (self.state, self.pos_obs[:, :, :self.args.pos]), False)
            action_env_multi[1:] = action_env_teacher[1:]

        if self.continuous:
            action_env_multi = [self.wrap_action(action_env_multi[i], self.action_high[i], self.action_low[i])
                                for i in range(self.num_agents)]
        #
        if self.args.train_mode == 2:
            self.obs.append(self.state.cpu())

        # model return action_env_multi, entropy, log_prob
        state_multi, reward_multi, self.done, self.info = self.env.step(list(action_env_multi))
        # add to buffer
        self.reward_org = reward_multi.copy()
        if 'clip' in self.args.clip:
            reward_multi = reward_multi.clip(-1, 1)
        if self.args.norm_reward:
            for id, rd in enumerate(reward_multi):
                reward_multi[id] = self.norm_r[min(id, 2)].update(reward_multi[id])
        self.reward = torch.tensor(reward_multi).float().to(self.device)
        self.state = torch.from_numpy(state_multi).float().to(self.device)
        if 'Multi' in self.args.env:
            self.pos_obs = torch.from_numpy(self.info['Pose_Obs']).float().to(self.device)
            in_area = torch.from_numpy(self.info['in_area']).long().to(self.device)
            self.in_labels.append(in_area)
        self.eps_len += 1
        self.values.append(value_multi)
        self.entropies.append(entropy)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward.unsqueeze(1))
        self.R_preds.append(R_pred)
        return self

    def action_test(self, determinstic=False):

        with torch.no_grad():
            value_multi, action_env_multi, entropy, log_prob, (R_pred, in_pred) = self.model(
                (Variable(self.state), self.pos_obs[:, :, :self.args.pos]), determinstic)
        if 'attack' in self.args.aux and self.num_agents > 1:
                value_teacher, action_env_teacher, _, _, _ = self.teacher(
                (self.state, self.pos_obs[:, :, :self.args.pos]), False)
                action_env_multi[1:] = action_env_teacher[1:]
        if 'rule' in self.args.tracker_net:
            action_env_multi[0] = self.rule_actor.act(self.env.bbox_pred)
        if self.continuous:
            action_env_multi = [self.wrap_action(action_env_multi[i], self.action_high[i], self.action_low[i])
                                for i in range(self.num_agents)]

        state_multi, self.reward, self.done, self.info = self.env.step(list(action_env_multi))
        self.state = torch.from_numpy(state_multi).float().to(self.device)
        reward_multi = torch.tensor(self.reward).float().to(self.device)
        if 'Multi' in self.args.env:
            self.pos_obs = torch.from_numpy(self.info['Pose_Obs']).float().to(self.device)
            relative_pos = self.info['Relative_Pose'][0, 1]
            relative_pos[0] -= 250
            self.relative_errors.append(np.fabs(relative_pos))
            self.mislead_steps.append(self.info['metrics']['mislead'])
            self.viewed_steps.append(self.info['metrics']['target_viewed'])
            self.viewed_dist.append(self.info['metrics']['d_in'])
            self.rewards.append(self.reward)
        self.eps_len += 1

        return self

    def reset(self):
        state_np = self.env.reset()
        self.state = torch.from_numpy(state_np).float().to(self.device)
        self.num_agents = self.state.shape[0]
        self.eps_len = 0
        self.eps_num += 1

        self.pos_obs = torch.zeros(self.num_agents, max(self.num_agents, 2), self.args.pos).to(self.device)
        self.relative_pos = torch.Tensor([250, 0]).float().to(self.device)
        self.reset_rnn_hiden()
        if 'rule' in self.args.tracker_net:
            self.rule_actor.reset(self.env.bbox_pred)
        self.force_ctl = False
        self.count_lost = 0
        self.mislead_steps = []
        self.viewed_dist = []
        self.viewed_steps = []
        self.relative_errors = []
        self.rewards = []


    def action_sample(self):
        with torch.no_grad():
            value_multi, action_env_multi, entropy, log_prob, (R_pred, in_pred) = self.model(
                    (Variable(self.state), self.pos_obs[:, :, :self.args.pos]))
            if 'rule' in self.args.tracker_net:
                action_env_multi[0] = self.rule_actor.act(self.env.bbox_pred)
            if 'teacher' in self.args.aux:
                    value_teacher, action_env_teacher, _, _, _ = self.teacher(
                        (self.state, self.pos_obs[:, :, :self.args.pos]), False)
                    if self.force_ctl:
                        action_env_multi[0] = action_env_teacher[0]

                    if self.num_agents > 1:  # teacher target
                        action_env_multi[1:] = action_env_teacher[1:]

        if self.continuous:
            action_env_multi = [self.wrap_action(action_env_multi[i], self.action_high[i], self.action_low[i])
                                for i in range(self.num_agents)]
        # add to buffer
        self.obs.append(self.state[:1].cpu())
        prob_to_save = self.teacher.probs[:1].cpu()
        onehot_teacher = torch.zeros_like(prob_to_save).scatter_(1, torch.argmax(prob_to_save, 1, keepdim=True), 1)
        self.action_prob_eps.append(onehot_teacher)
        # self.pos_obs_eps.append(self.pos_obs[:1])
        self.pos_obs_eps.append(self.relative_pos[0])
        if 'Multi' in self.args.env or 'Maze' in self.args.env:
            if self.eps_len == 0:
                in_area = torch.from_numpy(np.array([1])).long()
                perfect = 1
            else:
                in_area = torch.from_numpy(self.info['in_area']).long()
                perfect = self.info['metrics']['perfect']
            self.in_labels.append(in_area.cpu())
            self.perfects.append(perfect)

        # update environment
        state_multi, self.reward, self.done, self.info = self.env.step(list(action_env_multi))
        self.n_steps += 1

        if self.info['metrics']['target_viewed'] == 0:
            self.count_lost += 1
        else:
            self.count_lost = 0

        # add to buffer
        reward_multi = torch.tensor(self.reward).float().to(self.device)
        self.state = torch.from_numpy(state_multi).float().to(self.device)
        if 'Multi' in self.args.env:
            self.pos_obs = torch.from_numpy(self.info['Pose_Obs']).float().to(self.device)
            self.relative_pos = torch.Tensor(self.info['Relative_Pose'][0, 1]).float().to(self.device)

        self.eps_len += 1
        self.rewards.append(reward_multi.unsqueeze(1).cpu())

        return self

    def save_buffer(self):
        save_dir = os.path.join(self.args.mem, "%02d" % self.rank, 'tmp', "%04d" % self.eps_num + '.pt')
        torch.save({'obs': self.obs,
                    'rewards': self.rewards,
                    'in_labels': self.in_labels,
                    'perfects': self.perfects,  # for sampling start point
                    'action_prob': self.action_prob_eps,
                    'pos_obs': self.pos_obs_eps,
                    'length': self.eps_len},
                    save_dir
                    )
        self.action_prob_eps = []
        self.rewards = []
        self.in_labels = []
        self.perfects = []
        self.obs = []
        self.pos_obs_eps = []
        return save_dir

    def save_evaluation(self, path):
        save_dir = os.path.join(path, "%04d" % self.eps_num + '.pt')
        torch.save({'rewards': self.rewards,
                    'distractors': self.viewed_dist,
                    'tracked': self.viewed_steps,
                    'pos_obs': self.pos_obs_eps,
                    'mislead': self.mislead_steps,
                    'errors': self.relative_errors,
                    'length': self.eps_len},
                    save_dir
                    )
        self.rewards = []
        self.mislead_steps = []
        self.viewed_dist = []
        self.viewed_steps = []
        self.relative_errors = []
        self.pos_obs_eps = []
        return save_dir

    def clean_buffer(self, done):
        # gt
        self.rewards = []
        self.in_labels = []
        self.obs_tracker = []
        self.values = []
        self.entropies = []
        self.log_probs = []
        self.R_preds = []
        self.in_preds = []
        if done:
            # save the recent episodes, should long enough
            if 'replay' in self.args.aux:
                in_labels_eps = torch.stack(self.in_labels_eps).cpu()
                num_good = (in_labels_eps[-100:] == 0).sum()
                num_lost = (in_labels_eps[-100:] == 1).sum()
                num_mislead = (in_labels_eps[-100:] == 2).sum()
                if num_good > 95:
                    class_name = 'good'
                elif num_mislead > num_lost:
                    class_name = 'mislead'
                else:
                    class_name = 'lost'
                save_dir = os.path.join(self.args.mem, class_name, "%02d" % self.rank + "%02d" % (self.eps_num%100) +'.pt')
                torch.save({'obs': self.obs_tracker_eps,
                            'rewards': self.rewards_eps,
                            'in_labels': self.in_labels_eps},
                           save_dir
                           )
            # clean
            self.obs_tracker_eps = []
            self.rewards_eps = []
            self.in_labels_eps = []

        return self

    def reset_rnn_hiden(self, ):
        self.model.reset_hiden(self.num_agents)
        if 'teacher' in self.args.aux or 'attack' in self.args.aux:
            self.teacher.reset_hiden(self.num_agents)

    def update_rnn_hiden(self):
        self.model.update_hiden()
        if 'teacher' in self.args.aux or 'attack' in self.args.aux:
            self.teacher.update_hiden()

    def init_memory(self):
        if self.args.mem is not None:
            self.dir_memory = os.path.join(self.args.mem)
            if not os.path.exists(self.dir_memory):
                os.mkdir(self.dir_memory)
            if not os.path.exists(os.path.join(self.dir_memory, "%02d" % self.rank)):
                os.mkdir(os.path.join(self.dir_memory, "%02d" % self.rank))
            if not os.path.exists(os.path.join(self.dir_memory, "%02d" % self.rank, 'tmp')):
                os.mkdir(os.path.join(self.dir_memory, "%02d" % self.rank, 'tmp'))

    def optimize(self, params, optimizer, shared_model, training_mode, device_share):
        R = torch.zeros(self.num_agents, 1).to(self.device)
        if not self.done:
            # predict value
            state = self.state
            value_multi, _, _, _, _ = self.model(
                (Variable(state, requires_grad=True), self.pos_obs[:, :, :self.args.pos]))
            for i in range(self.num_agents):
                R[i][0] = value_multi[i].data
        self.values.append(Variable(R).to(self.device))
        policy_loss = torch.zeros(self.num_agents, 1).to(self.device)
        value_loss = torch.zeros(self.num_agents, 1).to(self.device)
        entropies = torch.zeros(self.num_agents, self.dim_action).to(self.device)
        w_entropies = float(self.args.entropy_others)*torch.ones(self.num_agents, self.dim_action).to(self.device)
        w_entropies[0][:] = float(self.args.entropy)
        if self.num_agents > 1:
            w_entropies[1][:] = float(self.w_entropy_target)
        R = Variable(R, requires_grad=True).to(self.device)
        gae = torch.zeros(1, 1).to(self.device)
        l1_loss = L1Loss(reduction='none')

        for i in reversed(range(len(self.rewards))):
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

        lr_weight = torch.ones_like(value_loss)
        self.model.zero_grad()
        loss_tracker = lr_weight[0]*(policy_loss[0] + 0.5 * value_loss[0]).mean()

        if self.num_agents > 1:
            loss_target = lr_weight[1]*(policy_loss[1] + 0.5 * value_loss[1]).mean()
        else:
            loss_target = 0
        
        if self.num_agents > 2:
            loss_others = (lr_weight[2:]*(policy_loss[2:] + 0.5 * value_loss[2:])).mean()
        else:
            loss_others = 0
        
        if training_mode == 0:  # train tracker
            loss = loss_tracker
        elif training_mode == 1:  # train adv group
            loss = loss_target + loss_others
        else:
            loss = loss_tracker + loss_target + loss_others

        aux_loss = []
        if 'reward' in self.args.aux:
            max_num = self.R_preds[0].shape[0]
            if training_mode == 1:
                pred_loss = l1_loss(torch.stack(self.R_preds)[:, 1:], torch.stack(self.rewards)[:, 1:max_num, :])
            elif training_mode == 0:
                pred_loss = l1_loss(torch.stack(self.R_preds)[:, :1], torch.stack(self.rewards)[:, :1, :])
            else:
                pred_loss = l1_loss(torch.stack(self.R_preds), torch.stack(self.rewards)[:, :max_num, :])
            pred_loss = pred_loss.sum()
            loss += pred_loss
            aux_loss.append(pred_loss)

        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(params, 10)
        ensure_shared_grads(self.model, shared_model, self.device, device_share)
        optimizer.step()
        # move short buffer to long buffer
        self.clean_buffer(self.done)

        return policy_loss, value_loss, entropies, aux_loss, lr_weight


class reward_normalizer(object):
    def __init__(self):
        self.reward_mean = 0
        self.reward_std = 1
        self.num_steps = 0
        self.vk = 0
    def update(self, reward):
        self.num_steps += 1
        if self.num_steps == 1:
            self.reward_mean = reward
            self.vk = 0
            self.reward_std = 1
        else:
            delt = reward - self.reward_mean
            self.reward_mean = self.reward_mean + delt/self.num_steps
            self.vk = self.vk + delt * (reward-self.reward_mean)
            self.reward_std = np.sqrt(self.vk/(self.num_steps - 1))
        reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return reward


class rule_ctl(object):
    def __init__(self, shape):
        self.init_bbox = None
        self.init_center = shape[0]/2.0

    def reset(self, init_bbox):
        # self.init_center = init_bbox[0] + init_bbox[2]/2
        self.init_size = init_bbox[2] * init_bbox[3]

    def act(self, bbox):
        size = bbox[2] * bbox[3]
        center = bbox[0] + bbox[2] / 2
        error_center = (center - self.init_center) / float(self.init_center)
        error_size = (size - self.init_size) / float(self.init_size)
        action = 6

        if abs(error_center) < 0.1:
            if error_size > 0.2:  # backward
                action = 1
            elif abs(error_size) < 0.2:  # stop
                action = 6
            elif error_size < -0.2:  # forward
                action = 0
        elif 0.1 <= error_center <= 0.3:
            action = 2
        elif -0.3 <= error_center <= -0.1:
            action = 3
        elif error_center > 0.3:
            action = 4
        elif error_center < -0.3:
            action = 5

        return action
