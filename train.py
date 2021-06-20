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
from random import choice, randint
from utils_basic import load_opponent

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
    else:
        device = torch.device('cpu')

    # if args.gpu >= 0:
    device_share = torch.device('cpu')

    if env == None:
        env = create_env(env_name, args, rank)

    params = shared_model.parameters()
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(params, lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=args.lr)

    player = Agent(None, env, args, None, device)
    player.rank = rank
    player.init_memory()
    player.w_entropy_target = args.entropy_target
    player.gpu_id = gpu_id

    # prepare model
    player.model = build_model(
        player.env.observation_space, player.env.action_space, args, device)
    player.model = player.model.to(device)
    player.model.train()
    if args.fix:
        env.seed(args.seed)
    else:
        env.seed(randint(0, args.seed))
    player.reset()
    reward_sum = torch.zeros(player.num_agents).to(device)
    reward_sum_org = np.zeros(player.num_agents)
    ave_reward = np.zeros(2)
    ave_reward_longterm = np.zeros(2)
    count_eps = 0
    if args.old is not None:
        opponent_list = os.listdir(args.old)

    while True:
        # sys to the shared model
        player.model.load_state_dict(shared_model.state_dict())

        if player.done:
            if args.fix:
                env.seed(args.seed)
            else:
                env.seed(randint(0, args.seed))
            player.reset()
            reward_sum = torch.zeros(player.num_agents).to(device)
            reward_sum_org = np.zeros(player.num_agents)
            count_eps += 1
            if args.old is not None:
                # update saved opponent
                saved_opponent = load_opponent(os.path.join(args.old, choice(opponent_list)))
                player.teacher.load_state_dict(saved_opponent, strict=False)

        player.update_rnn_hiden()
        t0 = time.time()
        for i in range(args.num_steps):
            player.action_train()
            reward_sum += player.reward
            reward_sum_org += player.reward_org
            if player.done:
                for i, r_i in enumerate(reward_sum):
                    if i ==2:
                        writer.add_scalar('train/reward_' + str(i), reward_sum[2:].sum(), player.n_steps)
                        if args.norm_reward:
                            writer.add_scalar('train/reward_org_' + str(i), reward_sum_org[2:].sum(), player.n_steps)
                        break
                    else:
                        writer.add_scalar('train/reward_' + str(i), r_i, player.n_steps)
                        if args.norm_reward:
                            writer.add_scalar('train/reward_org_' + str(i), reward_sum_org[i].sum(), player.n_steps)
                break
        fps = i / (time.time() - t0)

        training_mode = train_modes[rank]
        policy_loss, value_loss, entropies, pred_loss, lr_weight = player.optimize(params, optimizer, shared_model, training_mode, device_share)

        for i in range(min(player.num_agents, 3)):
            writer.add_scalar('train/policy_loss_'+str(i), policy_loss[i], player.n_steps)
            writer.add_scalar('train/value_loss_'+str(i), value_loss[i], player.n_steps)
            writer.add_scalar('train/entropies'+str(i), entropies[i], player.n_steps)
            writer.add_scalar('train/lr_weights' + str(i), lr_weight[i], player.n_steps)
        for i in range(len(pred_loss)):
            writer.add_scalar('train/pred_R_loss_'+str(i), pred_loss[i], player.n_steps)
        writer.add_scalar('train/ave_reward', ave_reward[0] - ave_reward_longterm[0], player.n_steps)
        writer.add_scalar('train/mode', training_mode, player.n_steps)
        writer.add_scalar('train/fps', fps, player.n_steps)

        n_iters[rank] = player.n_steps

        if train_modes[rank] == -100:
            env.close()
            break