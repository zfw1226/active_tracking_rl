from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from environment import create_env
from model import build_model
from player_util import Agent
import logging
import numpy as np
from utils_basic import cv2_show, setup_logger
import cv2
import json
from tqdm import trange

def save(file_path, obj):
    item = os.path.split(file_path)
    dic = item[0]
    file = item[1]
    if not os.path.exists(dic):
        os.makedirs(dic)
    if '.png' in file:
        cv2.imwrite(file_path, obj)
    elif '.json' in file:
        with open(file_path, 'w') as f:
            json.dump(obj, f)
# add end

parser = argparse.ArgumentParser(description='A3C_EVAL')
parser.add_argument('--env', default='UnrealTrackingUrbancity-DiscreteColorGoal-v0', metavar='ENV', help='environment to train on')
parser.add_argument('--num-episodes', type=int, default=100,metavar='NE', help='how many episodes in evaluation')
parser.add_argument('--load-model-dir', default=None, metavar='LMD', help='folder to load trained models')
parser.add_argument('--load-teacher', default=None, metavar='LCD', help='folder to load trained teacher models')
parser.add_argument('--load-tracker', default=None, metavar='LCD', help='folder to load trained tracker models')
parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument('--csv', default=None, metavar='SV', help='write to csv')
parser.add_argument('--save', default=None, metavar='SV', help='folder to save imgs')
parser.add_argument('--save-eval', default=None, metavar='SV', help='folder to save imgs')
parser.add_argument('--save-reset', default=None, metavar='SV', help='folder to save reset imgs')
parser.add_argument('--render', dest='render', action='store_true', help='render test')
parser.add_argument('--render-freq', type=int, default=1, metavar='RF', help='Frequency to watch rendered game play')
parser.add_argument('--max-test-length', type=int, default=100000, metavar='M', help='maximum length of an episode')
parser.add_argument('--model', default='pos-act-lstm-novision', metavar='M', help='config the overall model')
parser.add_argument('--tracker-net', default='tiny-ConvLSTM-att-lstm-layer', metavar='M', help='config tracker network')
parser.add_argument('--stack-frames', type=int, default=1, metavar='SF', help='Choose whether to stack observations')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
parser.add_argument('--gpu-id', type=int, default=-1, help='GPU to use [-1 CPU only] (default: -1)')
parser.add_argument('--obs', default='img', metavar='UE', help='unreal env')
parser.add_argument('--clip', dest='clip_reward', action='store_true', help='clip reward')
parser.add_argument('--gray', dest='gray', action='store_true', help='gray image')
parser.add_argument('--crop', dest='crop', action='store_true', help='crop image')
parser.add_argument('--inv', dest='inv', action='store_true', help='inverse image')
parser.add_argument('--flip', dest='flip', action='store_true', help='flip image and action')
parser.add_argument('--rescale', dest='rescale', action='store_true', help='rescale image to [-1, 1]')
parser.add_argument('--normalize', dest='normalize', action='store_true', help='normalize image')
parser.add_argument('--input-size', type=int, default=80, metavar='IS', help='input image size')
parser.add_argument('--rnn-out', type=int, default=128, metavar='LO', help='rnn output size')
parser.add_argument('--rnn-teacher', type=int, default=128, metavar='LO', help='rnn output size for teacher model')
parser.add_argument('--fuse-out', type=int, default=128, metavar='LO', help='fuse output size')
parser.add_argument('--pos', type=int, default=5, metavar='LO', help='the length of the pos vector')
parser.add_argument('--single', dest='single', action='store_true', help='single agent')
parser.add_argument('--aux', default='none', metavar='M', help='Model type to use')
parser.add_argument('--mem', default=None, metavar='M', help='save memory')
parser.add_argument('--noise', type=int, default=0, metavar='LO', help='noise type')
parser.add_argument('--channel', dest='channel', action='store_true', help='random channel order')
parser.add_argument('--pytrack', default='/home/ubuntu/codes/pytracking', metavar='M', help='path to pytracking')
parser.add_argument('--pytrack-model', default='dimp', metavar='M', help='model')
parser.add_argument('--pytrack-net', default='dimp18', metavar='M', help='network')
parser.add_argument('--early-done', dest='early_done', action='store_true', help='early stop the episode')

if __name__ == '__main__':
    args = parser.parse_args()

    torch.set_default_tensor_type('torch.FloatTensor')

    log = {}
    setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
        args.log_dir, args.env))
    log['{}_mon_log'.format(args.env)] = logging.getLogger(
        '{}_mon_log'.format(args.env))

    gpu_id = args.gpu_id

    if gpu_id >= 0:
        torch.manual_seed(args.seed)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    d_args = vars(args)
    for k in d_args.keys():
        log['{}_mon_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    env = create_env("{}".format(args.env), args)
    env.seed(args.seed)
    num_tests = 0
    reward_total_sum = 0
    eps_success = 0
    rewards_his = []
    len_lis = []
    player = Agent(None, env, args, None, device)
    player.model = build_model(
        env.observation_space, env.action_space, args, device)
    player.model.to(device)
    player.gpu_id = gpu_id

    if args.load_model_dir is not None:
        saved_state = torch.load(
            args.load_model_dir,
            map_location=lambda storage, loc: storage)
        if args.load_model_dir[-3:] == 'pth':
            player.model.load_state_dict(saved_state['model'], strict=False)
        else:
            player.model.load_state_dict(saved_state)

    if args.load_tracker is not None:
        saved_state = torch.load(
            args.load_tracker, map_location=lambda storage, loc: storage)
        saved_tracker = saved_state['model']
        saved_tracker = {name: param for name, param in saved_tracker.items() if
                        'tracker' in name and 'critic' not in name}

        player.model.load_state_dict(saved_tracker, strict=False)

    if args.save is not None:
        if not os.path.exists(args.save):
            os.mkdir(args.save)
    
    if args.save_reset is not None:
        if not os.path.exists(args.save_reset):
            os.mkdir(args.save_reset)

    if args.save_eval is not None:
        folder = os.path.join(args.save_eval)
        if not os.path.exists(folder):
            os.mkdir(folder)
        folder = os.path.join(args.save_eval, args.env)
        if not os.path.exists(folder):
            os.mkdir(folder)
        folder = os.path.join(args.save_eval, args.env, args.tracker_net)
        if not os.path.exists(folder):
            os.mkdir(folder)
        folder = os.path.join(args.save_eval, args.env, args.tracker_net)
        if not os.path.exists(folder):
            os.mkdir(folder)
    with trange(args.num_episodes) as t:
        player.model.eval()
        freq_list = []
        mislead_num = 0
        lost_num = 0
        for i_episode in trange(args.num_episodes):
            player.env.seed(args.seed)
            player.reset()
            reward_sum = np.zeros(player.num_agents)
            if args.save_reset is not None:
                img = env.render(mode='rgb_array')
                cv2.imwrite(os.path.join(args.save_reset, str(i_episode) + '_reset.png'), img)
            while True:
                if args.render:
                    if 'VizDoom' in args.env:
                        cv2_show(env)
                    elif 'Unreal' in args.env:
                        # env.render()
                        cv2_show(env, False)
                    else:
                        env.render()
                if args.save is not None:
                    if not os.path.exists(os.path.join(args.save, str(i_episode))):
                        os.mkdir(os.path.join(args.save, str(i_episode)))
                    img = env.render(mode='rgb_array')
                    cv2.imwrite(os.path.join(args.save, str(i_episode), "%05d" % player.eps_len +'.png'), img)

                player.action_test(False)
                reward_sum += player.reward

                if player.done:
                    num_tests += 1
                    rewards_his.append(reward_sum[:2])
                    len_lis.append(player.eps_len)
                    reward_mean = np.array(rewards_his).mean(0)
                    reward_std = np.array(rewards_his).std(0)
                    len_mean = np.array(len_lis).mean()
                    len_std = np.array(len_lis).std()
                    freq_list.append(np.sum(player.viewed_dist))
                    if np.sum(player.viewed_steps[-10:]) < 3:  # the target is not observed at the end
                        if np.sum(player.mislead_steps[-5:]) > 3:
                            mislead_num += 1.0
                        else:
                            lost_num += 1.0
                    elif player.eps_len >= 500:
                        eps_success += 1

                    success_rate = eps_success / num_tests
                    mislead_rate = mislead_num / num_tests
                    lost_rate = lost_num / num_tests
                    if args.save_eval is not None:
                        folder = os.path.join(args.save_eval, args.env, args.tracker_net, str(args.seed))
                        if not os.path.exists(folder):
                            os.mkdir(folder)
                        player.save_evaluation(folder)
                    t.set_postfix(Reward=reward_mean[0], E_Length=len_mean, S_rate=success_rate)
                    break
        log['{}_mon_log'.format(args.env)].info(
            "R_mean: {0:.2f}, R_std: {1:.2f}, EL_mean: {2:.2f}, EL_std {3:.2f}, S_rate: {4:.2f}, Freq: {5:.2f}, MisLead: {6:.2f}, Lost: {7:.2f}".format(
                float(reward_mean[0]), float(reward_std[0]), len_mean, len_std, float(success_rate),
                float(np.sum(freq_list)/np.sum(len_lis)), float(mislead_rate), float(lost_rate)))
        # write to csv
        if args.load_teacher is None:
            args.load_teacher = 'noteacher'
        header = ['Env', 'Seed', 'Teacher', 'R_mean', 'R_std', 'EL_mean', 'EL_std', 'S_rate', 'Freq', 'Mislead', 'Lost']
        rows = [{'Env': args.env, 'Seed': args.seed, 'Teacher': args.load_teacher[-7:],
                 'R_mean': float(reward_mean[0]), 'R_std': float(reward_std[0]), 'EL_mean': len_mean,'EL_std': len_std,
                 'S_rate': float(success_rate), 'Freq': float(np.sum(freq_list)/np.sum(len_lis)), 'Mislead':float(mislead_rate), 'Lost': float(lost_rate),
                 },
                ]
        if args.csv is not None:
            import csv
            if not os.path.exists(args.csv):
                with open(args.csv, 'w') as f:
                    f_csv = csv.DictWriter(f, header)
                    f_csv.writeheader()
                    f_csv.writerows(rows)
            else:
                with open(args.csv, 'a') as f:
                    f_csv = csv.DictWriter(f, header)
                    f_csv.writerows(rows)
        player.env.close()
        os._exit(0)
    # except KeyboardInterrupt:
    #     print("Shutting down")
    #     player.env.close()
    #     os._exit(0)