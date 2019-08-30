from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
from environment import create_env
from utils import setup_logger
from player_util import Agent
import time
import logging
from tensorboardX import SummaryWriter
import os
from model import build_model
from utils import cv2_show


def test(args, shared_model, train_modes, n_iters):
    ptitle('Test Agent')
    n_iter = 0
    writer = SummaryWriter(os.path.join(args.log_dir, 'Test'))
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env),
                 r'{0}/logger'.format(args.log_dir))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    if args.env_base is None:
        env = create_env(args.env, args)
    else:
        env = create_env(args.env_base, args)
    env.seed(args.seed)
    start_time = time.time()
    count_eps = 0

    player = Agent(None, env, args, None, device)
    player.gpu_id = gpu_id
    player.model = build_model(
        player.env.observation_space, player.env.action_space, args, device).to(device)
    player.model.eval()
    max_score = -100
    seed = args.seed
    last_iter = 0
    iter_th = args.init_step
    while True:
        reward_sum = np.zeros(2)
        len_sum = 0
        for i_episode in range(args.test_eps):
            player.model.load_state_dict(shared_model.state_dict())
            player.env.seed(seed)
            # seed += 1
            player.reset()
            reward_sum_ep = np.zeros(player.num_agents)
            fps_counter = 0
            t0 = time.time()
            count_eps += 1
            while True:
                if args.render:
                    if 'Unreal' in args.env:
                        cv2_show(env, False)
                    else:
                        env.render()
                player.action_test()
                fps_counter += 1
                reward_sum_ep += player.reward
                if player.done:
                    reward_sum += reward_sum_ep[:2]
                    len_sum += player.eps_len
                    fps = fps_counter / (time.time()-t0)
                    n_iter = 0
                    for n in n_iters:
                        n_iter += n

                    for rank in range(len(n_iters)):
                        if n_iter < args.init_step:
                            train_modes[rank] = 0
                        elif args.train_mode == 2 and n_iter - last_iter > iter_th:
                            train_modes[rank] = 1 - train_modes[rank]
                            last_iter = n_iter
                            iter_th = args.init_step if train_modes[rank]==0 else args.adv_step
                        else:
                            train_modes[rank] = args.train_mode

                    for i, r_i in enumerate(reward_sum_ep):
                        writer.add_scalar('test/reward'+str(i), r_i, n_iter)

                    writer.add_scalar('test/fps', fps, n_iter)
                    writer.add_scalar('test/eps_len', player.eps_len, n_iter)
                    break

        ave_reward_sum = reward_sum/args.test_eps
        len_mean = len_sum/args.test_eps
        reward_step = reward_sum / len_sum
        log['{}_log'.format(args.env)].info(
            "Time {0}, ave eps reward {1}, ave eps length {2}, reward step {3}".
            format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                ave_reward_sum, len_mean, reward_step))

        # save model
        if ave_reward_sum[0] >= max_score:
            print('Save best!')
            max_score = ave_reward_sum[0]
            model_dir = os.path.join(args.log_dir, 'all-best-{0}.dat'.format(n_iter))
            tracker_model_dir = os.path.join(args.log_dir, 'tracker-best.dat'.format(n_iter))
            target_model_dir = os.path.join(args.log_dir, 'target-best.dat'.format(n_iter))
        else:
            model_dir = os.path.join(args.log_dir, 'all-new.dat'.format(args.env))
            tracker_model_dir = os.path.join(args.log_dir, 'tracker-new.dat')
            target_model_dir = os.path.join(args.log_dir, 'target-new.dat')

        torch.save(player.model.state_dict(), model_dir)
        if args.split:
            torch.save(player.model.player0.state_dict(), tracker_model_dir)
            if not args.single:
                torch.save(player.model.player1.state_dict(), target_model_dir)

        time.sleep(args.sleep_time)
        if n_iter > args.max_step:
            env.close()
            for id in range(0, args.workers):
                train_modes[id] = -100
            break