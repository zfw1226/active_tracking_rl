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
from sampler import sampler
from shared_optim import SharedRMSprop, SharedAdam
import time
from datetime import datetime
from replayer import optimize_asys

#undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G', help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T', help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy', type=float, default=0.01, metavar='T', help='parameter for entropy (default: 0.01)')
parser.add_argument('--entropy-target', type=float, default=0.03, metavar='T', help='parameter for target entropy (default: 0.01)')
parser.add_argument('--entropy-others', type=float, default=0.03, metavar='T', help='parameter for distractors entropy (default: 0.03)')
parser.add_argument('--seed', type=int, default=2, metavar='S', help='random seed (default: 2)')
parser.add_argument('--seed-test', type=int, default=4, metavar='S', help='random seed for testing(default: 4)')
parser.add_argument('--workers', type=int, default=4, metavar='W', help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS', help='number of forward steps in A3C (default: 20)')
parser.add_argument('--test-eps', type=int, default=50, metavar='M', help='maximum length of an episode (default: 10000)')
parser.add_argument('--env', default='UnrealTrackMulti-FlexibleRoomAdv-DiscreteColor-v3', metavar='ENV', help='environment to train on')
parser.add_argument('--env-base', default='UnrealTrackMulti-FlexibleRoomAdv-DiscreteColor-v1', metavar='ENV', help='environment to test on')
parser.add_argument('--optimizer', default='Adam', metavar='OPT', help='shares optimizer choice of Adam or RMSprop')
parser.add_argument('--amsgrad', default=True, metavar='AM', help='Adam optimizer amsgrad parameter')
parser.add_argument('--load-model-dir', default=None, metavar='LMD', help='folder to load trained full models from')
parser.add_argument('--load-tracker', default=None, metavar='LMD', help='folder to load trained tracker model from')
parser.add_argument('--load-teacher', default=None, metavar='LMD', help='folder to load trained teacher models from')
parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument('--model', default='pos-act-lstm-novision', metavar='M', help='config the overall model')
parser.add_argument('--tracker-net', default='tiny-ConvLSTM-att-lstm-layer', metavar='M', help='config tracker network')
parser.add_argument('--aux', default='none', metavar='M', help='Model type to use')
parser.add_argument('--gpu-ids', type=int, default=-1, nargs='+', help='GPUs to use [-1 CPU only] (default: -1) for distributed workers')
parser.add_argument('--gpu', type=int, default=-1, metavar='LO', help='GPU device for centralized training')
parser.add_argument('--obs', default='img', metavar='UE', help='unreal env')
# preprocess and augmentation
parser.add_argument('--clip', default='no', metavar='C', help='how to clip reward')
parser.add_argument('--gray', dest='gray', action='store_true', help='gray image')
parser.add_argument('--crop', dest='crop', action='store_true', help='crop image')
parser.add_argument('--inv', dest='inv', action='store_true', help='inverse image')
parser.add_argument('--rescale', dest='rescale', action='store_true', help='rescale image to [-1, 1]')
parser.add_argument('--norm-reward', dest='norm_reward', action='store_true', help='normalize image')
parser.add_argument('--render', dest='render', action='store_true', help='render test')
parser.add_argument('--fix', dest='fix', action='store_true', help='fix random seed')
parser.add_argument('--channel', dest='channel', action='store_true', help='random channel order')
parser.add_argument('--flip', dest='flip', action='store_true', help='flip image')
parser.add_argument('--single', dest='single', action='store_true', help='single agent env')
parser.add_argument('--shared-optimizer', dest='shared_optimizer', action='store_true', help='use an optimizer without shared statistics.')
parser.add_argument('--load-optimizer', dest='load_optimizer', action='store_true', help='load an optimizer.')
parser.add_argument('--train-mode', type=int, default=-1, metavar='TM', help='his')
parser.add_argument('--stack-frames', type=int, default=1, metavar='SF', help='Choose number of observations to stack')
parser.add_argument('--input-size', type=int, default=120, metavar='IS', help='input image size')
parser.add_argument('--rnn-out', type=int, default=128, metavar='LO', help='lstm output size')
parser.add_argument('--rnn-teacher', type=int, default=128, metavar='LO', help='lstm output size')
parser.add_argument('--fuse-out', type=int, default=128, metavar='LO', help='fuse output size')
parser.add_argument('--pos', type=int, default=5, metavar='LO', help='the length of the pos vector')
parser.add_argument('--sleep-time', type=int, default=15, metavar='LO', help='seconds')
parser.add_argument('--max-step', type=int, default=2000, metavar='LO', help='max learning (k)steps')
parser.add_argument('--init-step', type=int, default=2000, metavar='LO', help='initial sampling steps')
parser.add_argument('--buffer-size', type=int, default=200, metavar='LO', help='max buffer size')
parser.add_argument('--batch-size', type=int, default=8, metavar='LO', help='batch size')
parser.add_argument('--mem', default=None, metavar='M', help='maze-lstm-pos')
parser.add_argument('--old', default=None, metavar='M', help='old opponent')
parser.add_argument('--pytrack', default='none', metavar='M', help='path to pytracking')
parser.add_argument('--pytrack-model', default='dimp', metavar='M', help='model')
parser.add_argument('--pytrack-net', default='dimp18', metavar='M', help='network')
parser.add_argument('--early-done', dest='early_done', action='store_true', help='early stop the episode')

if __name__ == '__main__':

    args = parser.parse_args()
    args.shared_optimizer = True
    if args.gpu_ids == -1 and args.gpu == -1:
        torch.manual_seed(args.seed)
        args.gpu_ids = [-1]
    else:
        if args.gpu_ids == -1:
            args.gpu_ids = [-1]
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn', force=True)
    device_share = torch.device('cpu')

    env = create_env(args.env, args)

    shared_model = build_model(
        env.observation_space, env.action_space, args, device_share).to(device_share)
    shared_model.share_memory()

    if args.load_model_dir is not None:
        saved_state = torch.load(
            args.load_model_dir,
            map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state['model'], strict=False)
        # optimizer_state = saved_state['optimizer']
  
    if args.load_tracker is not None:
        if args.load_tracker[-3:] == 'pth':
            saved_state = torch.load(
                args.load_tracker,
                map_location=lambda storage, loc: storage)
            saved_tracker = saved_state['model']

            saved_tracker = {name: param for name, param in saved_tracker.items() if
                            'tracker' in name}

        else:
            saved_tracker = torch.load(args.load_tracker)
            saved_tracker = {name: param for name, param in saved_tracker.items() if
                           'tracker' in name}

        shared_model.load_state_dict(saved_tracker, strict=False)
    params = shared_model.parameters()
    if args.shared_optimizer:
        print('share memory')
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(params, lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(params, lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    if args.load_model_dir is not None and args.load_optimizer:
        if args.load_model_dir[-3:] == 'pth':
            optimizer.load_state_dict(optimizer_state)
            print('Load previous optimizer')

    current_time = datetime.now().strftime('%b%d_%H-%M')
    args.log_dir = os.path.join(args.log_dir, args.env, current_time)

    if args.workers == -1: # for debuging
        if args.train_mode == 6:
            sampler(0, args, shared_model, [], [])
        else:
            train(0, args, shared_model, optimizer, [], [])
    else:
        processes = []
        manager = mp.Manager()
        train_modes = manager.list()
        n_iters = manager.list()

        p = mp.Process(target=test, args=(args, shared_model, optimizer, train_modes, n_iters))
        p.start()
        processes.append(p)
        time.sleep(args.sleep_time)

        for rank in range(0, args.workers):
            dirs_list = manager.list()
            if args.train_mode >= 6:
                p = mp.Process(target=sampler, args=(
                    rank, args, shared_model, train_modes, n_iters))
            else:
                p = mp.Process(target=train, args=(
                    rank, args, shared_model, optimizer, train_modes, n_iters))
            p.start()
            processes.append(p)
            time.sleep(args.sleep_time)

        if args.train_mode == 6:
            p = mp.Process(target=optimize_asys, args=(optimizer, shared_model, device_share, args, train_modes, n_iters))
            p.start()
            processes.append(p)

        for p in processes:
            time.sleep(args.sleep_time)
            p.join()






