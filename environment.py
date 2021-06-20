from __future__ import division
import gym
import numpy as np
from collections import deque
# import gym_real
from cv2 import resize
from gym.spaces.box import Box
import cv2
from random import choice
import sys
import gym_unrealcv

def create_env(env_id, args, rank=-1):
    env = gym.make(env_id)
    print ('build env')
    if args.single:
      env = listspace(env)
    if args.rescale is True:
        env = Rescale(env, args)
    if 'img' in args.obs:
        env = UnrealPreprocess(env, args)

    env = frame_stack(env, args)  # (n) -> (stack, n) // (c, w, h) -> (stack, c, w, h)
    env = pytracker(env, args)
    return env


class Rescale(gym.Wrapper):
    def __init__(self, env, args):
        super(Rescale, self).__init__(env)
        self.new_maxd = 1.0
        self.new_mind = -1.0
        self.mx_d = 255.0
        self.mn_d = 0.0
        shape = env.observation_space[0].shape
        self.num_agents = len(self.observation_space)
        self.observation_space = [Box(self.new_mind, self.new_maxd, shape) for i in range(self.num_agents)]

        self.obs_range = self.mx_d - self.mn_d
        self.args = args
        self.inv_img = self.choose_rand_seed() and self.args.inv

    def rescale(self, x):
        obs = x.clip(self.mn_d, self.mx_d)
        new_obs = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                    ) / self.obs_range) + self.new_mind
        return new_obs

    def reset(self):
        ob = self.env.reset()
        # rescale image to [-1, 1]
        ob = self.rescale(np.float32(ob))
        if self.args.channel:
            self.order_new = list(np.random.permutation(3))
            ob = ob[..., self.order_new]
        # invert image
        self.inv_img = self.choose_rand_seed() and self.args.inv
        if self.inv_img:
            ob = - ob
        return ob

    def step(self, action):
        ob, rew, done, info = self.env.step(action)

        ob = self.rescale(np.float32(ob))
        if self.args.channel:
            ob = ob[..., self.order_new]
        if self.inv_img:
            ob = - ob

        return ob, rew, done, info

    def choose_rand_seed(self):
        return choice([True, False])

class UnrealPreprocess(gym.ObservationWrapper):
    def __init__(self, env, args):
        gym.ObservationWrapper.__init__(self, env)

        self.gray = args.gray
        self.crop = args.crop
        self.input_size = args.input_size

        obs_shape = self.observation_space[0].shape
        num_agnets = len(self.observation_space)
        self.channel = obs_shape[2]
        if abs(obs_shape[0] - self.input_size) + abs(obs_shape[1] - self.input_size) == 0:
            self.resize = False
        else:
            self.resize = True
        if self.gray is True:
            self.channel = 1

        self.observation_space = [Box(-1.0, 1.0, [self.channel, self.input_size, self.input_size],
                                      dtype=np.uint8) for i in range(num_agnets)]

    def process_frame_ue(self, frame, size=80):
        frame = frame.astype(np.float32)
        if self.crop:
            shape = frame.shape
            frame = frame[:shape[0], int(shape[1] / 2 - shape[0] / 2): int(shape[1] / 2 + shape[0] / 2)]
        if self.resize:
            frame = resize(frame, (size, size))
        if self.gray:
            frame = frame.mean(2)  # color to gray
            frame = np.expand_dims(frame, 0)
        else:
            frame = frame.transpose(2, 0, 1)
        return frame

    def observation(self, observation):
        obses = []
        for obs in observation:
            obses.append(self.process_frame_ue(obs, self.input_size))
        return np.array(obses)


class frame_stack(gym.Wrapper):
    def __init__(self, env, args):
        super(frame_stack, self).__init__(env)
        self.stack_frames = args.stack_frames
        self.max_num_agents = len(self.observation_space)
        self.frames = [deque([], maxlen=self.stack_frames) for i in range(self.max_num_agents)]

    def reset(self):
        ob = self.env.reset()
        self.num_agents = len(ob)
        ob = np.float32(ob)
        for i in range(self.num_agents):
            for _ in range(self.stack_frames):
                self.frames[i].append(ob[i])
        return self.observation()

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        ob = np.float32(ob)
        for i in range(self.num_agents):
            self.frames[i].append(ob[i])
        ob = self.observation()
        if type(done) == list:
            done = all(done)
        return ob, rew, done, info

    def observation(self):
        ob = [np.stack(self.frames[i], axis=0) for i in range(self.num_agents)]
        return np.array(ob)

class pytracker(gym.Wrapper):
    def __init__(self, env, args):
        super(pytracker, self).__init__(env)
        self.siame = False
        if 'pytrack' in args.pytrack:
            self.use_pytrack = True
            sys.path.append(args.pytrack)
            from pytracking.evaluation import Tracker
            self.TrackClass = Tracker(args.pytrack_model, args.pytrack_net)
            self.vis_tracker = self.TrackClass.get_tracker()
        elif 'Siam' in args.pytrack:
            self.siame = True
            self.use_pytrack = True
            sys.path.append(args.pytrack)
            from DaSiamRPN.code.tracker import Tracker
            self.vis_tracker = Tracker()
        else:
            self.use_pytrack = False
        self.args = args
    
    def reset(self):
        ob = self.env.reset()
        try:
            while self.env.mask_percent < 0.02:
                print('invalid mask')
                ob = self.env.reset()
            self.env.set_action_factors(np.array([np.random.uniform(0.8, 1.2), np.random.uniform(0.7, 1.2)]), 0.6)
            if self.args.early_done:
                self.env.set_early_stop(True)
            else:
                self.env.set_early_stop(False)
        except:
            pass
        self.score = None
        if self.use_pytrack:
            optional_box = self.env.bbox_init[0]
            frame = self.env.img_color
            if self.siame:
                self.vis_tracker.initialize(frame, optional_box)
                self.bbox_pred = self.vis_tracker.track(frame)
            else:
                self.vis_tracker.initialize(frame, self.TrackClass.init_bbox(optional_box))
                if 'rule' in self.args.tracker_net:
                    self.vis_tracker.fix_area(False)
                else:
                    self.vis_tracker.fix_area(True)
                out = self.vis_tracker.track(frame)
                self.bbox_pred = [int(s) for s in out['target_bbox'][1]]
                self.score = out['score'][1].clone().detach()
        return ob

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self.score = None
        if self.use_pytrack:
            frame = self.env.img_color
            frame_disp = frame.copy()
            if self.siame:
                res = self.vis_tracker.track(frame)
                self.bbox_pred = res
                if self.args.render:
                    # plot GT for visualization
                    bbox = self.env.bbox
                    cv2.rectangle(frame_disp, (int(bbox[0]), int(bbox[1])),
                                  (int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])), (0, 0, 255), 5)
                    cv2.rectangle(frame, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
                    cv2.imshow('SiamRPN', frame)
                    cv2.waitKey(1)
            else:
                out = self.vis_tracker.track(frame)
                info['pytrack_res'] = out
                self.bbox_pred = [int(s) for s in out['target_bbox'][1]]
                self.score = out['score'][1].clone().detach()
                if self.args.render:
                    # plot GT for visualization
                    bbox = self.env.bbox
                    cv2.rectangle(frame_disp, (int(bbox[0]), int(bbox[1])),
                                  (int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])), (0, 0, 255), 5)
                    self.TrackClass.plot(frame_disp, out)

        return ob, rew, done, info

class listspace(gym.Wrapper):
    def __init__(self, env):
        super(listspace, self).__init__(env)
        if type(self.observation_space) == list:
            self.num_agents = len(self.observation_space)
        else:
            self.observation_space = [self.observation_space]
            self.action_space = [self.action_space]
            self.num_agents = 1

    def reset(self):
        ob = self.env.reset()
        return [ob]

    def step(self, action):
        ob, rew, done, info = self.env.step(action[0])
        return [ob], [rew], done, info