import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import gym
from gym import spaces
from gym.utils import seeding
from gym_track2d.envs.generators import RandomMazeGenerator, RandomBlockMazeGenerator
from gym_track2d.envs.navigator import Navigator, RamAgent


class Track1v1Env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 maze_type='U',
                 pob_size=6,
                 action_type='VonNeumann',
                 obs_type='Partial',
                 target_mode='PZR',
                 live_display=True,
                 render_trace=True,
                 level=0):
        """Initialize the maze. DType: list"""
        # Random seed with internal gym seeding
        self.seed()
        self.num_agents_max = self.num_agents = 2
        self.maze_type = maze_type
        self.level = level
        # Size of the partial observable window
        self.pob_size = pob_size
        # Maze: 0: free space, 1: wall
        self.init_maze(self.maze_type)

        self.render_trace = render_trace
        self.traces = []
        self.traces_relative = []
        self.action_type = action_type
        self.obs_type = obs_type
        self.target_mode = target_mode

        # If True, show the updated display each time render is called rather
        # than storing the frames and creating an animation at the end
        self.live_display = live_display

        self.state = None

        # Action space
        tracker_action_space = self.define_action(self.action_type)
        target_action_space = self.define_action(self.action_type)
        self.action_space = [tracker_action_space, target_action_space]

        # Observation space
        print(self.obs_type)
        tracker_obs_space = self.define_observation(self.obs_type)
        target_obs_space = self.define_observation(self.obs_type)
        self.observation_space = [tracker_obs_space, target_obs_space]

        # nav
        self.Target = []
        for i in range(self.num_agents-1):
            if 'Nav' in self.target_mode:
                random_th = 0
                self.Target.append(Navigator(self.action_space[i+1], self.maze_generator, random_th))
            if 'Ram' in self.target_mode:
                self.Target.append(RamAgent(self.action_space[i+1]))

        # Colormap: order of color is, free space, wall, agent, food, poison
        self.cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray', 'yellow'])
        self.bounds = [0, 1, 2, 3, 4, 5, 6]  # values for each color
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
        self.C_step = 0

    def step(self, action):
        # Player 0: try to catch player 1
        # Player 1: try to reach the goal and avoid player 0
        old_state = self.state.copy()
        # Update current state
        rewards = np.zeros(self.num_agents)
        done = False
        action = list(action)
        # move agents
        for i in range(self.num_agents - 1):
            if 'Ram' in self.target_mode:
                action[i+1] = self.Target[i].step()
            if 'Nav' in self.target_mode:
                action[i+1], goal = self.Target[i].step(old_state[i + 1], self.maze_generator, None)

        for i in range(self.num_agents):
            self.state[i], self.C_collision[i] = self._next_state(self.state[i], int(action[i]),
                                                                  self.action_type)

        self.traces_relative = []
        for j in range(self.num_agents):
            self.traces_relative.append([np.array(self.init_states[i]) - np.array(self.init_states[j]) for i in
                                         range(self.num_agents)])
        d_all = np.array([np.linalg.norm(np.array(self.state[i]) - np.array(self.state[0])) for i in range(self.num_agents)])

        max_distance = float(self.pob_size)
        distance = d_all[1]

        r_track = 1 - 2*distance/max_distance
        r_track = max(r_track, -1)  # [-1, 1]
        if 'PZR' in self.target_mode:
            r_target = -r_track - max(distance - max_distance, 0)/max_distance
            r_target = max(r_target, -1)
        else:
            r_target = -r_track
        rewards[0] = r_track
        rewards[1] = r_target

        if distance <= max_distance:
            self.C_far = 0
        else:
            self.C_far += 1
        if self.C_far > 10:
            done = True

        self.C_reward += rewards
        self.C_step += 1

        # Additional info
        info = {}
        self.distance = info['distance'] = d_all[1]
        # Footprint: Record agent trajectory
        self.traces.append(self.state[1])
        obs = self._get_obs()
        info['traces'] = self.traces
        info['traces_relative'] = self.traces_relative
        if 'Nav' in self.target_mode or 'Ram' in self.target_mode:
            obs = obs[:2]
            rewards = rewards[:2]
        return obs, rewards, done, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def reset(self):
        # Reset maze
        self.init_maze(self.maze_type)
        self.state = self.init_states
        if 'Nav' in self.target_mode:
            for i in range(self.num_agents-1):
                self.Target[i].reset(self.init_states[i+1], self.goal_states[i+1], self.maze_generator)
        if 'Ram' in self.target_mode:
            for i in range(self.num_agents-1):
                self.Target[i].reset()
        self.distance = np.sum(np.abs(np.array(self.state[0]) - np.array(self.state[1])))
        self.C_reward = np.zeros(self.num_agents)
        self.C_step = 0
        self.C_collision = np.zeros(self.num_agents)
        self.C_far = 0

        # Clean the list of ax_imgs, the buffer for generating videos
        self.ax_imgs = []
        # Clean the traces of the trajectory
        self.traces = [self.init_states[0]]
        self.traces_relative = [np.array(self.init_states[i]) - np.array(self.init_states[0]) for i in range(self.num_agents)]
        obs = self._get_obs()
        if 'Nav' in self.target_mode or 'Ram' in self.target_mode:
            obs = obs[:2]
        return obs

    def render(self, mode='human', close=False):
        import time
        time.sleep(0.03)
        if close:
            plt.close()
            return

        obs = self._get_full_obs()
        partial_obs = self._get_partial_obs(0, self.pob_size)

        # For rendering traces: Only for visualization, does not affect the observation data
        if self.render_trace:
            obs[list(zip(*self.traces[:-1]))] = 6

        # Create Figure for rendering
        if not hasattr(self, 'fig'):  # initialize figure and plotting axes
            self.fig, (self.ax_full, self.ax_partial) = plt.subplots(nrows=1, ncols=2)
        self.ax_full.axis('off')
        self.ax_partial.axis('off')

        self.fig.show()
        if self.live_display:
            # Only create the image the first time
            if not hasattr(self, 'ax_full_img'):
                self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            if not hasattr(self, 'ax_partial_img'):
                self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)
            # Update the image data for efficient live video
            self.ax_full_img.set_data(obs)
            self.ax_partial_img.set_data(partial_obs)
        else:
            # Create a new image each time to allow an animation to be created
            self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)

        plt.draw()

        if self.live_display:
            # Update the figure display immediately
            self.fig.canvas.draw()
        else:
            # Put in AxesImage buffer for video generation
            self.ax_imgs.append([self.ax_full_img, self.ax_partial_img])  # List of axes to update figure frame

            self.fig.set_dpi(200)

        return self.fig

    def init_maze(self, maze_type):
        if maze_type == 'Random':
            if self.level > 0:
                r = self.level * 0.02
            else:
                r = .03*np.random.random()
            self.maze_generator = RandomMazeGenerator(width=80, height=80, complexity=r, density=r)
        elif maze_type == 'Block':
            if self.level > 0:
                r = self.level * 0.05
            else:
                r = 0.15*np.random.random()
            self.maze_generator = RandomBlockMazeGenerator(maze_size=80, obstacle_ratio=r)
        elif maze_type == 'Empty':
            self.maze_generator = RandomBlockMazeGenerator(maze_size=80, obstacle_ratio=0)
        self.maze = np.array(self.maze_generator.get_maze())
        self.maze_size = self.maze.shape
        self.init_states = self.maze_generator.sample_close_states(self.num_agents, 1)
        self.goal_states = self.maze_generator.sample_goal(self.num_agents)
        while self.goal_test(self.init_states[0]):  # Goal check
            self.goal_states = self.maze_generator.sample_goal(self.num_agents)

    def define_action(self, action_type):
        if action_type == 'VonNeumann':  # Von Neumann neighborhood
            num_actions = 4
        elif action_type == 'Moore':  # Moore neighborhood
            num_actions = 8
        else:
            raise TypeError('Action type must be either \'VonNeumann\' or \'Moore\'')
        return spaces.Discrete(num_actions)

    def define_observation(self, obs_type):
        low_obs = 0  # Lowest integer in observation
        high_obs = 6  # Highest integer in observation
        if obs_type == 'Full':
            obs_space = spaces.Box(low=low_obs,
                                                high=high_obs,
                                                shape=(1, self.maze_size[0], self.maze_size[1]),
                                                )
        elif self.obs_type == 'Partial':
            obs_space = spaces.Box(low=low_obs,
                                           high=high_obs,
                                           shape=(1, self.pob_size * 2 + 1, self.pob_size * 2 + 1),
                                           dtype=np.float32
                                           )
        else:
            raise TypeError('Observation type must be either \'full\' or \'partial\'')
        return obs_space

    def goal_test(self, state):
        """Return True if current state is a goal state."""
        if type(self.goal_states[0]) == list:
            return list(state) in self.goal_states
        elif type(self.goal_states[0]) == tuple:
            return tuple(state) in self.goal_states

    def _next_state(self, state, action, action_type='VonNeumann'):
        """Return the next state from a given state by taking a given action."""

        # Transition table to define movement for each action
        if action_type == 'VonNeumann':
            transitions = {0: [-1, 0], 1: [+1, 0], 2: [0, -1], 3: [0, +1]}
        elif action_type == 'Moore':
            transitions = {0: [-1, 0], 1: [+1, 0], 2: [0, -1], 3: [0, +1],
                           4: [-1, +1], 5: [+1, +1], 6: [-1, -1], 7: [+1, -1]}

        new_state = [state[0] + transitions[action][0], state[1] + transitions[action][1]]
        if self.maze[new_state[0]][new_state[1]] == 1:  # Hit wall, stay there
            return state, True
        else:  # Valid move for 0, 2, 3, 4
            return new_state, False

    def _get_obs(self):
        if self.obs_type == 'Full':
            obs = [np.expand_dims(self._get_full_obs(), 0) for i in range(self.num_agents)]
            return np.array(obs)
        elif self.obs_type == 'Partial':
            obs = [np.expand_dims(self._get_partial_obs(i, self.pob_size), 0) for i in range(self.num_agents)]
            return np.array(obs)

    def _get_full_obs(self):
        """Return a 2D array representation of maze."""
        obs = np.array(self.maze)

        # Set current position
        for i in range(self.num_agents):
            if i < 2:
                color = 2+2*i
            else:
                color = 2 + np.random.randint(0, 3)
            obs[self.state[i][0]][self.state[i][1]] = color

        return obs

    def _get_partial_obs(self, id=0, size=1, vec=False):
        """Get partial observable window according to Moore neighborhood"""
        # Get maze with indicated location of current position and goal positions
        maze = self._get_full_obs()
        maze[self.state[id][0]][self.state[id][1]] = 2+2*id
        pos = np.array(self.state[id])

        under_offset = np.min(pos - size)
        over_offset = np.min(len(maze) - (pos + size + 1))
        offset = np.min([under_offset, over_offset])

        if offset < 0:  # Need padding
            maze = np.pad(maze, np.abs(offset), 'constant', constant_values=1)
            pos += np.abs(offset)
        maze_p = maze[pos[0] - size: pos[0] + size + 1, pos[1] - size: pos[1] + size + 1]
        if vec:
            maze_p = maze_p.reshape(self.v_len)
        return maze_p