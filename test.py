from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
from environment import create_env
from utils_basic import setup_logger, check_path, check_disk
from player_util import Agent
import time
import logging
from tensorboardX import SummaryWriter
import os
from model import build_model

def test(args, shared_model, optimizer, train_modes, n_iters):
    ptitle('Test Agent')
    n_iter = 0
    writer = SummaryWriter(os.path.join(args.log_dir, 'Test'))
    gpu_id = args.gpu
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
    env.seed(args.seed_test)
    start_time = time.time()
    count_eps = 0

    player = Agent(None, env, args, None, device)
    player.gpu_id = gpu_id
    player.model = build_model(
        player.env.observation_space, player.env.action_space, args, device).to(device)
    player.model.eval()
    max_score = -100
    seed = args.seed_test
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
            fps_all = []
            while True:
                if args.render:
                    if 'Unreal' in args.env:
                        env.render()
                        # cv2_show(env, False)
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

                    for i, r_i in enumerate(reward_sum_ep):
                        writer.add_scalar('test/reward'+str(i), r_i, n_iter)

                    fps_all.append(fps)
                    writer.add_scalar('test/fps', fps, n_iter)
                    writer.add_scalar('test/eps_len', player.eps_len, n_iter)
                    break

        ave_reward_sum = reward_sum/args.test_eps
        len_mean = len_sum/args.test_eps
        reward_step = reward_sum / len_sum
        log['{}_log'.format(args.env)].info(
            "Time {0}, ave eps reward {1}, ave eps length {2}, reward step {3}, FPS {4}".
            format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                np.around(ave_reward_sum, decimals=2), np.around(len_mean, decimals=2),
                np.around(reward_step, decimals=2), np.around(np.mean(fps_all), decimals=2)))

        # save model
        if args.train_mode == 1:
            new_score = ave_reward_sum[1]
        else:
            new_score = ave_reward_sum[0]
        if new_score >= max_score:
            print('save best!')
            max_score = new_score
            model_dir = os.path.join(args.log_dir, 'best-{}.pth'.format(str(n_iter//10000)))
        else:
            # model_dir = os.path.join(args.log_dir, 'new.pth')
            model_dir = os.path.join(args.log_dir, 'new-{}.pth'.format(str(n_iter//10000)))
        state_to_save = {"model": player.model.state_dict()}
        torch.save(state_to_save, model_dir)

        time.sleep(args.sleep_time)
        if n_iter/1000 > args.max_step or check_disk('./') > 95:
            print('Reach Max Step or Disk is full')
            for id in range(len(train_modes)):
                train_modes[id] = -100
            # env.close()
            break