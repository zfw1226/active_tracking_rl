from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import os
import time
import torch.nn as nn
import random
import shutil
import time

def get_meomory_list(memory_dir):
    l = os.listdir(os.path.join(memory_dir))
    return l

def load_tmp(dir_memory):
    mem_list = get_meomory_list(os.path.join(dir_memory, 'tmp'))
    total_eps = len(mem_list)
    mem_buffer = []
    steps = 0
    if total_eps == 0:
        return mem_buffer, total_eps, steps
    for name in mem_list:
        tmp_dir = os.path.join(dir_memory, 'tmp', name)
        try:
            tmp_dict = torch.load(tmp_dir)
            os.remove(tmp_dir)
            steps += tmp_dict['length']
            mem_buffer.append(tmp_dict)
        except:
            continue

    return mem_buffer, total_eps, steps

def optimize_asys(shared_optimizer, shared_model, device_share, args, train_modes, n_iters):
    # collect a number of samples before optimize
    while True:
        time.sleep(0.1)
        if len(n_iters) == args.workers and min(n_iters) > args.init_step:
            break
    print('Start optimize')
    writer = SummaryWriter(os.path.join(args.log_dir, 'Asys'))
    ptitle('Asys-Training')
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(args.gpu))
    import pickle
    model = pickle.loads(pickle.dumps(shared_model))

    model = model.to(device)

    params = model.tracker.parameters()

    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(params, lr=args.lr)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, amsgrad=args.amsgrad)

    kl_loss = torch.nn.KLDivLoss(reduction='sum')
    mem_buffer = []
    total_eps = 0
    total_step = 0
    n_steps = 0
    batch_size = args.batch_size
    start_time = time.time()
    model.zero_grad()
    epoch = 0
    last_print_eps = 0
    while True:
        epoch += 1
        # update samples
        samplers = os.listdir(args.mem)
        mem_new = []
        num_eps_new = 0
        num_steps_new = 0
        for sample in samplers:
            mem, num_eps, steps = load_tmp(os.path.join(args.mem, sample))
            if steps == 0:
                continue
            mem_new += mem
            num_eps_new += num_eps
            num_steps_new += steps
        mem_buffer += mem_new
        total_eps += num_eps_new
        total_step += num_steps_new
        if total_eps > args.buffer_size:
            mem_buffer = mem_buffer[-args.buffer_size:]
        # training
        loss_total = 0
        model.train()

        samples = random.sample(mem_buffer, batch_size * min(2, int(len(mem_buffer) / batch_size)))  
        
        # init samples
        stop_step = [[0] for i in range(batch_size)]
        obs_batch = [[] for i in range(batch_size)]
        action_prob_batch = [[] for i in range(batch_size)]
        pos_batch = [[] for i in range(batch_size)]
        perfect_batch = [[] for i in range(batch_size)]
        min_length = [0 for i in range(batch_size)]

        for i, tmp_dict in enumerate(samples):
            index = i % batch_size
            if min(min_length) > 500:
                break
            if min_length[index] > 500:
                continue
            stop_step[index].append(stop_step[index][-1] + tmp_dict['length'])
            obs_batch[index] += tmp_dict['obs']
            action_prob_batch[index] += tmp_dict['action_prob']
            pos_batch[index] += tmp_dict['pos_obs']
            perfect_batch[index] += tmp_dict['perfects']
            min_length[index] = stop_step[index][-1]
            obs_seq = torch.stack(tmp_dict['obs'])

        stop_step = np.array(stop_step)
        min_length = min(min(min_length), 300)
        for index in range(batch_size):
            # random sample start point
            max_length = len(perfect_batch[index])
            candidates_start = np.where(np.array(perfect_batch[index]) == 1)[0]
            candidates_start = candidates_start[np.where(candidates_start + min_length <= max_length)]
            start_id = int(np.random.choice(candidates_start))  # other sample policy
            end_id = int(start_id + min_length)
            obs_seq = torch.stack(obs_batch[index])
            obs_batch[index] = obs_seq[start_id:end_id]
            action_prob_batch[index] = torch.stack(action_prob_batch[index][start_id:end_id])

        obs_batch = torch.stack(obs_batch)
        action_prob_batch = torch.stack(action_prob_batch) 

        if args.flip:
            obs_batch_flip = torch.flip(obs_batch, [-1])
            obs_batch = torch.cat([obs_batch, obs_batch_flip])
            action_prob_batch_flip = action_prob_batch[..., [0, 1, 3, 2, 5, 4, 6]]
            action_prob_batch = torch.cat([action_prob_batch, action_prob_batch_flip])
        batch_size_train = obs_batch.shape[0]
        model.tracker.reset_internal(device, batch_size_train)

        loss_kl = 0
        for step_n in range(min_length):  # forward the one epoch step by step
            # optimize
            obs = obs_batch[:, step_n, 0, 0, :]
            obs = Variable(obs, requires_grad=True).to(device)

            action, entropy, log_prob, prob = model.tracker(obs, test=True, critic=False)
            gt = action_prob_batch[:, step_n, 0].to(device)
            l1 = kl_loss(prob.log(), gt)
            loss_kl += l1

            if step_n % args.num_steps == 0 or step_n in stop_step:
                loss_kl = loss_kl / batch_size_train
                loss = loss_kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 20)
                optimizer.step()
                model.zero_grad()
                loss_total += float(loss)
                writer.add_scalar('train/loss', float(loss_kl), n_steps)
                loss_kl = 0
                n_steps += 1
                model.tracker.update_internal(np.where(stop_step == step_n)[0])

        if total_eps - last_print_eps > args.buffer_size:
            last_print_eps = total_eps
            print(' Epoch:{0}, Time:{1}, Loss:{2:.4}, Total_Eps:{3}, Total_Steqs:{4}'.format(
                epoch, time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), loss_total / min_length, total_eps, total_step))
        shared_model.load_state_dict(model.state_dict())
        if train_modes[-1] == -100:
            break