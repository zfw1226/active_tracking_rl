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
import torch.nn as nn
from random import choice, randint
from utils_basic import load_opponent


def sampler(rank, args, shared_model, train_modes, n_iters):
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

    env = create_env(env_name, args, rank)

    if args.fix:
        env.seed(args.seed)
    else:
        env.seed(randint(0, args.seed))
    player = Agent(None, env, args, None, device)
    player.rank = rank
    player.init_memory()
    player.w_entropy_target = args.entropy_target
    player.gpu_id = gpu_id

    # prepare model
    player.model = build_model(
        player.env.observation_space, player.env.action_space, args, device)
    player.model = player.model.to(device)
    player.model.eval()

    player.reset()
    count_eps = 0
    if args.old is not None:
        opponent_list = os.listdir(args.old)
        saved_opponent = load_opponent(os.path.join(args.old, choice(opponent_list)))
    while True:
            player.model.load_state_dict(shared_model.state_dict())
            if args.fix:
                env.seed(args.seed)
            else:
                env.seed(randint(0, args.seed))
            player.reset()
            reward_sum_ep = np.zeros(player.num_agents)
            fps_counter = 0
            t0 = time.time()
            count_eps += 1
            fps_all = []
            while True:
                player.action_sample()
                fps_counter += 1
                reward_sum_ep += player.reward
                if player.done:
                    fps = fps_counter / (time.time()-t0)
                    saved_pth = player.save_buffer()
                    if args.old is not None:
                        saved_opponent = load_opponent(os.path.join(args.old, choice(opponent_list)))
                        player.teacher.load_state_dict(saved_opponent, strict=False)

                    for i, r_i in enumerate(reward_sum_ep):
                        if i == 2:
                            writer.add_scalar('train/reward_' + str(i), reward_sum_ep[2:].sum(), player.n_steps)
                            break
                        else:
                            writer.add_scalar('train/reward_' + str(i), r_i, player.n_steps)
                    writer.add_scalar('train/mode', training_mode, player.n_steps)
                    writer.add_scalar('train/fps', fps, player.n_steps)
                    writer.add_scalar('train/eps_len', player.eps_len, player.n_steps)
                    n_iter += player.eps_len
                    n_iters[rank] = n_iter
                    fps_all.append(fps)
                    break

            training_mode = train_modes[rank]

            if train_modes[rank] == -100:
                env.close()
                break