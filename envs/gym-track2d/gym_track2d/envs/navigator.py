from gym_track2d.envs.Astar_solver import AstarSolver
import numpy as np


class Navigator(object):
    def __init__(self, action_space, maze_generator, random_th=0.):
        self.all_actions = list(range(action_space.n))
        self.maze_generator = maze_generator
        self.random_th = random_th

    def step(self, state, maze_generator, ref_goal=None):
        count_res = 0
        planB = False
        # new goal
        if self._goal_test(state) or self.a_i>=len(self.plan_actions):  # Reach Goal or actions is done
            if ref_goal==None or np.random.random() > self.random_th:
                self.goal_states = maze_generator.sample_goal(1)[0]
            else:
                self.goal_states = ref_goal
            self.planner = AstarSolver(state, self.all_actions, self.maze, self.goal_states)
            # check path
            while self.planner.solvable() is False or len(self.planner.get_actions()) < 1:
                count_res += 1
                if count_res > 5:
                    planB = True
                    break
                if ref_goal == None:
                    self.goal_states = maze_generator.sample_goal(1)[0]
                else:
                    self.goal_states = ref_goal
                self.planner = AstarSolver(state, self.all_actions, self.maze, self.goal_states)

            if planB:
                self.plan_actions = np.random.choice(self.all_actions, 10)
            else:
                self.plan_actions = self.planner.get_actions()

            self.a_i = 0
        action = self.plan_actions[self.a_i]
        self.a_i += 1
        return action, self.goal_states

    def reset(self, init_state, goal_state, maze_generator):
        self.state = self.init_states = init_state
        self.goal_states = goal_state
        self.maze = maze_generator.get_maze()
        self.planner = AstarSolver(self.init_states, self.all_actions, self.maze, self.goal_states)
        count_res = 0
        planB = False
        # check reachable
        while self.planner.solvable() is False or len(self.planner.get_actions()) < 1:
            count_res += 1
            if count_res > 5:
                planB = True
                break
            self.goal_states = maze_generator.sample_goal(1)[0]
            self.planner = AstarSolver(self.init_states, self.all_actions, self.maze, self.goal_states)
        if planB:
            self.plan_actions = np.random.choice(self.all_actions, 10)
        else:
            self.plan_actions = self.planner.get_actions()
        self.a_i = 0
        return self.state

    def _goal_test(self, state):
        """Return True if current state is a goal state."""
        if type(self.goal_states[0]) == list:
            return list(state) in self.goal_states
        elif type(self.goal_states[0]) == tuple:
            return tuple(state) in self.goal_states


class RamAgent(object):
    def __init__(self, action_space):
        self.all_actions = list(range(action_space.n))

    def step(self):
        action = self.plan_actions[self.a_i]
        self.a_i += 1
        if self.a_i >= len(self.plan_actions):
            if np.random.choice([0, 1], 1) == 0:
                action = np.random.choice(self.all_actions, 1)
                self.plan_actions = np.ones(np.random.randint(1, 10)) * action
            else:
                self.plan_actions = np.random.choice(self.all_actions, np.random.randint(1, 10))
            self.a_i = 0

        return action

    def reset(self):
        self.plan_actions = np.random.choice(self.all_actions, np.random.randint(1, 10))
        self.a_i = 0
        return self.plan_actions[0]
