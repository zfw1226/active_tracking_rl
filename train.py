from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
import torch.optim as optim
from environment import create_env

from model import build_model
from player_util import Agent
from tensorboardX import SummaryWriter
import os
import time


def train(rank, args, shared_model, optimizer, train_modes, n_iters, env=None):
    n_iter = 0
    writer = SummaryWriter(os.path.join(args.log_dir, 'Agent:{}'.format(rank)))
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    training_mode = args.train_mode
    env_name = args.env

    train_modes.append(training_mode)
    n_iters.append(n_iter)

    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
        device = torch.device('cuda:' + str(gpu_id))
        device_share = torch.device('cuda:' + str(args.gpu_ids[-1]))
    else:
        device = device_share = torch.device('cpu')
    if env is None:
        env = create_env(env_name, args)

    if args.train_mode == 0:
        params = shared_model.player0.parameters()
    elif args.train_mode == 1:
        params = shared_model.player1.parameters()
    else:
        params = shared_model.parameters()
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(params, lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=args.lr)

    env.seed(args.seed)
    player = Agent(None, env, args, None, device)
    player.w_entropy_target = args.entropy_target
    player.gpu_id = gpu_id

    # prepare model
    player.model = build_model(
        player.env.observation_space, player.env.action_space, args, device)
    player.model = player.model.to(device)
    player.model.train()

    player.reset()
    reward_sum = torch.zeros(player.num_agents).to(device)
    reward_sum_org = np.zeros(player.num_agents)
    ave_reward = np.zeros(2)
    ave_reward_longterm = np.zeros(2)
    count_eps = 0
    while True:
        # sys to the shared model
        player.model.load_state_dict(shared_model.state_dict())

        if player.done:
            player.reset()
            reward_sum = torch.zeros(player.num_agents).to(device)
            reward_sum_org = np.zeros(player.num_agents)
            count_eps += 1

        player.update_rnn_hiden()
        t0 = time.time()
        for i in range(args.num_steps):
            player.action_train()
            reward_sum += player.reward
            reward_sum_org += player.reward_org
            if player.done:
                for j, r_i in enumerate(reward_sum):
                    writer.add_scalar('train/reward_' + str(j), r_i, player.n_steps)
                break
        fps = i / (time.time() - t0)

        # cfg training mode
        # 0: tracker 1: target -1:joint all
        training_mode = train_modes[rank]

        policy_loss, value_loss, entropies, pred_loss = player.optimize(params, optimizer, shared_model, training_mode, device_share)

        for i in range(min(player.num_agents, 3)):
            writer.add_scalar('train/policy_loss_'+str(i), policy_loss[i].mean(), player.n_steps)
            writer.add_scalar('train/value_loss_'+str(i), value_loss[i], player.n_steps)
            writer.add_scalar('train/entropies'+str(i), entropies[i].mean(), player.n_steps)
        writer.add_scalar('train/pred_R_loss', pred_loss, player.n_steps)
        writer.add_scalar('train/ave_reward', ave_reward[0] - ave_reward_longterm[0], player.n_steps)
        writer.add_scalar('train/mode', training_mode, player.n_steps)
        writer.add_scalar('train/fps', fps, player.n_steps)

        n_iter += 1
        n_iters[rank] = n_iter

        if train_modes[rank] == -100:
            env.close()
            break