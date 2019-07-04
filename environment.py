from __future__ import division
import gym
import numpy as np
from collections import deque
from cv2 import resize
from gym.spaces.box import Box
from random import choice


def create_env(env_id, args):
    if '2D' in env_id:
        import gym_track2d
    if 'Unreal' in env_id:
        import gym_unrealcv
    env = gym.make(env_id)
    # config observation pre-processing
    if args.normalize is True:
        env = NormalizedObs(env)
    if args.rescale is True:
        env = Rescale(env, args)  # rescale, inv
    if 'img' in args.obs and '2D' not in env_id:
        env = UnrealPreprocess(env, args)  # gray, crop, resize

    env = frame_stack(env, args)  # (n) -> (stack, n) // (c, w, h) -> (stack, c, w, h)

    return env


class Rescale(gym.Wrapper):
    def __init__(self, env, args):
        super(Rescale, self).__init__(env)
        self.new_maxd = 1.0
        self.new_mind = -1.0
        if type(env.observation_space) == list:
            self.mx_d = 255.0
            self.mn_d = 0.0
            shape = env.observation_space[0].shape
            self.num_agents = len(self.observation_space)
            self.observation_space = [Box(self.new_mind, self.new_maxd, shape) for i in range(self.num_agents)]
        else:
            self.mx_d = env.observation_space.high
            self.mn_d = env.observation_space.low
            shape = env.observation_space.shape
            self.observation_space = Box(self.new_mind, self.new_maxd, shape)
        self.obs_range = self.mx_d - self.mn_d
        self.args = args
        self.inv_img = self.choose_rand_seed() and self.args.inv

    def rescale(self, x):
        obs = x.clip(self.mn_d, self.mx_d)
        new_obs = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                    ) / self.obs_range) + self.new_mind
        return new_obs

    def _reset(self):
        ob = self.env.reset()

        # invert image
        self.inv_img = self.choose_rand_seed() and self.args.inv
        if self.inv_img:
            ob = 255 - ob

        # rescale image to [-1, 1]
        ob = self.rescale(np.float32(ob))
        return ob

    def _step(self, action):
        ob, rew, done, info = self.env.step(action)
        if self.inv_img:
            ob = 255 - ob
        ob = self.rescale(np.float32(ob))
        return ob, rew, done, info

    def choose_rand_seed(self):
        return choice([True, False])


class UnrealPreprocess(gym.ObservationWrapper):
    def __init__(self, env, args):
        gym.ObservationWrapper.__init__(self, env)

        self.gray = args.gray
        self.crop = args.crop
        self.input_size = args.input_size

        if type(self.observation_space) == list:
            obs_shape = self.observation_space[0].shape
            num_agnets = len(self.observation_space)
        else:
            obs_shape = self.observation_space.shape
            num_agnets = None
        self.channel = obs_shape[2]
        if abs(obs_shape[0] - self.input_size) + abs(obs_shape[1] - self.input_size) == 0:
            self.resize = False
        else:
            self.resize = True
        if self.gray is True:
            self.channel = 1
        if num_agnets is None:
            self.observation_space = Box(-1.0, 1.0, [self.channel, self.input_size, self.input_size], dtype=np.uint8)
        else:
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
        for i in range(len(observation)):
            obses.append(self.process_frame_ue(observation[i], self.input_size))
        return np.array(obses)


class frame_stack(gym.Wrapper):
    def __init__(self, env, args):
        super(frame_stack, self).__init__(env)
        self.stack_frames = args.stack_frames
        self.num_agents = len(self.observation_space)
        self.frames = [deque([], maxlen=self.stack_frames) for i in range(self.num_agents)]

    def reset(self):
        ob = self.env.reset()
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


class NormalizedObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))
        return (observation - unbiased_mean) / (unbiased_std + 1e-8)