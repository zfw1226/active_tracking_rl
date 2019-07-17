from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import create_env
from model import build_model
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
import time
from datetime import datetime


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (2D: 0.001, 3D: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G', help='discount factor for rewards (default: 0.9)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T', help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy', type=float, default=0.01, metavar='E', help='parameter for entropy(for tracker)')
parser.add_argument('--entropy-target', type=float, default=0.2, metavar='EC', help='parameter for entropy(for target)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--workers', type=int, default=1, metavar='W', help='how many training processes to use')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS', help='number of forward steps in A3C')
parser.add_argument('--test-eps', type=int, default=100, metavar='TE', help='maximum length of an episode')
parser.add_argument('--env', default='Track2D-BlockPartialPZR-v0', metavar='ENV', help='environment to train on')
parser.add_argument('--env-base', default='Track2D-BlockPartialNav-v0', metavar='ENVB', help='environment to test on ')
parser.add_argument('--optimizer', default='Adam', metavar='OPT', help='shares optimizer choice of Adam or RMSprop')
parser.add_argument('--amsgrad', default=True, metavar='AM', help='Adam optimizer amsgrad parameter')
parser.add_argument('--load-model-dir', default=None, metavar='LMD', help='folder to load trained models from')
parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument('--network', default='tat-maze-lstm', metavar='M', help='config Network Architecture')
parser.add_argument('--aux', default='reward', metavar='A', help='auxiliary task: reward/none')
parser.add_argument('--gpu-ids', type=int, default=-1, nargs='+', help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--obs', default='img', metavar='O', help='img or vector')
parser.add_argument('--single', dest='single', action='store_true', help='run on single agent env')
parser.add_argument('--gray', dest='gray', action='store_true', help='gray image')
parser.add_argument('--crop', dest='crop', action='store_true', help='crop image')
parser.add_argument('--inv', dest='inv', action='store_true', help='inverse image')
parser.add_argument('--rescale', dest='rescale', action='store_true', help='rescale image to [-1, 1]')
parser.add_argument('--render', dest='render', action='store_true', help='render image for visualization')
parser.add_argument('--shared-optimizer', dest='shared_optimizer', action='store_true', help='use a shared optimizer')
parser.add_argument('--split', dest='split', action='store_true', help='split model to save')
parser.add_argument('--train-mode', type=int, default=-1, metavar='TM', help='which agent to train(0:tracker 1:target)')
parser.add_argument('--stack-frames', type=int, default=1, metavar='SF', help='Choose number of observations to stack')
parser.add_argument('--input-size', type=int, default=80, metavar='IS', help='input image size')
parser.add_argument('--rnn-out', type=int, default=128, metavar='LO', help='rnn output size')
parser.add_argument('--sleep-time', type=int, default=0, metavar='ST', help='seconds to sleep after a process launched')
parser.add_argument('--max-step', type=int, default=150000, metavar='MS', help='max learning steps')
parser.add_argument('--init-step', type=int, default=-1, metavar='IS', help='steps not update target at beginning')
# Based on
# https://github.com/dgriff777/a3c_continuous

if __name__ == '__main__':

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        torch.manual_seed(args.seed)
        args.gpu_ids = [-1]
        device_share = torch.device('cpu')
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')
        if len(args.gpu_ids) > 1:
            device_share = torch.device('cpu')
        else:
            device_share = torch.device('cuda:' + str(args.gpu_ids[-1]))
    env = create_env(args.env, args)

    shared_model = build_model(
        env.observation_space, env.action_space, args, device_share).to(device_share)

    if args.train_mode == 0:
        params = shared_model.player0.parameters()
    elif args.train_mode == 1:
        params = shared_model.player1.parameters()
    else:
        params = shared_model.parameters()

    if args.load_model_dir is not None:
        saved_state = torch.load(
            args.load_model_dir,
            map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(params, lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(params, lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    current_time = datetime.now().strftime('%b%d_%H-%M')
    args.log_dir = os.path.join(args.log_dir, args.env, current_time)
    if args.gpu_ids[-1] == -1:
        env.close()

    processes = []
    manager = mp.Manager()
    train_modes = manager.list()
    n_iters = manager.list()
    p = mp.Process(target=test, args=(args, shared_model, train_modes, n_iters))
    p.start()
    processes.append(p)
    time.sleep(args.sleep_time)

    for rank in range(0, args.workers):
        p = mp.Process(target=train, args=(
            rank, args, shared_model, optimizer, train_modes, n_iters))
        p.start()
        processes.append(p)
        time.sleep(args.sleep_time)
    for p in processes:
        time.sleep(args.sleep_time)
        p.join()






