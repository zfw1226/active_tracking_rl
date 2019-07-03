import numpy as np
from numpy.random import randint
from itertools import product as cartesian_product
from skimage.draw import circle


class MazeGenerator(object):
    def __init__(self):
        self.maze = None
        self.static = False

    def static_goals(self):
        self.static = True
        shape = self.maze.shape
        self.candidate_goals = [[int(shape[0]/6), int(shape[1]/6)], [int(shape[0]*5/6), int(shape[1]/6)],
                                [int(shape[0] * 5 / 6), int(shape[1] * 5 / 6)], [int(shape[0]/6), int(shape[1]*5/6)]]
        for goal in self.candidate_goals:
            self.maze[goal[0]][goal[1]] = 0
        self.vector=0

    def sample_state(self, num=1):
        """Uniformaly sample an initial state and a goal state"""
        # Get indices for all free spaces, i.e. zero
        free_space = np.where(self.maze == 0)
        free_space = list(zip(free_space[0], free_space[1]))

        # Sample indices for initial state and goal state
        idxs = np.random.choice(len(free_space), size=num*2, replace=False)
        init_states = []
        goal_states = []
        # Convert initial state to a list, goal states to list of list
        for i in range(num):
            init_states.append(list(free_space[idxs[i]]))
            goal_states.append(list(free_space[idxs[i+num]]))
        
        return init_states, goal_states

    def sample_goal(self, num=1):
        goal_states = []
        if not self.static:
            np.random.seed()
            free_space = np.where(self.maze == 0)
            free_space = list(zip(free_space[0], free_space[1]))
            idxs = np.random.choice(len(free_space), size=num, replace=False)
            for i in range(num):
                goal_states.append(list(free_space[idxs[i]]))
        else:
            self.vector = (self.vector+1) % len(self.candidate_goals)
            goal_states.append(self.candidate_goals[self.vector])
        return goal_states

    def sample_close_states(self, num=1, max_distance=1):
        """Uniformaly sample initial states"""
        # Get indices for all free spaces, i.e. zero
        np.random.seed()
        free_space = np.where(self.maze == 0)
        free_space = list(zip(free_space[0], free_space[1]))

        # Sample indices for initial state and goal state
        idxs = np.random.choice(len(free_space), size=2, replace=False)
        init_states = []
        # Convert initial state to a list, goal states to list of list
        # state: tracker state_next : target
        if not self.static:
            state = list(free_space[idxs[0]])
        else:
            state = self.candidate_goals[0]
        init_states.append(state)
        state_next = self.get_around(state, max_distance)
        init_states.append(state_next)
        # find next state
        state_others, _ = self.sample_state(num-2)
        for i in range(num-2):
            init_states.append(state_others[i])

        return init_states
        
    def get_maze(self):
        return self.maze

    def get_around(self, state, max_distance):
        x_0 = max(0, state[0]-max_distance)
        x_1 = min(self.maze.shape[0]-1, state[0]+max_distance)
        y_0 = max(0, state[1]-max_distance)
        y_1 = min(self.maze.shape[1]-1, state[1]+max_distance)
        partial_maze = self.maze[x_0:x_1, y_0:y_1]
        partial_free_space = np.where(partial_maze == 0)
        partial_free_space = list(zip(partial_free_space[0], partial_free_space[1]))
        idxs = np.random.choice(len(partial_free_space), size=1, replace=False)
        state_next = list(partial_free_space[idxs[0]])
        state_next[0] += x_0
        state_next[1] += y_0
        return state_next


class SimpleMazeGenerator(MazeGenerator):
    def __init__(self, maze):
        super().__init__()
        
        self.maze = maze
        

class RandomMazeGenerator(MazeGenerator):
    def __init__(self, width=81, height=51, complexity=.75, density=.75):
        super().__init__()
        
        self.width = width
        self.height = height
        self.complexity = complexity
        self.density = density
        
        self.maze = self._generate_maze()
        
    def _generate_maze(self):
        """
        Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
        """
        # Only odd shapes
        shape = ((self.height // 2) * 2 + 1, (self.width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        self.complexity = int(self.complexity * (5 * (shape[0] + shape[1])))
        self.density    = int(self.density * ((shape[0] // 2) * (shape[1] // 2)))
        # Build actual maze
        Z = np.zeros(shape, dtype=bool)
        # Fill borders
        Z[0, :] = Z[-1, :] = 1
        Z[:, 0] = Z[:, -1] = 1
        # Make aisles
        for i in range(self.density):
            x, y = randint(0, shape[1]//2 + 1) * 2, randint(0, shape[0]//2 + 1) * 2
            Z[y, x] = 1
            for j in range(self.complexity):
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < shape[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < shape[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[randint(0, len(neighbours))]
                    if Z[y_, x_] == 0:
                        Z[y_, x_] = 1
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_
        return Z.astype(int)


class RandomBlockMazeGenerator(MazeGenerator):
    def __init__(self, maze_size, obstacle_ratio):
        super().__init__()
        
        self.maze_size = maze_size
        self.obstacle_ratio = obstacle_ratio
        
        self.maze = self._generate_maze()
        
    def _generate_maze(self):
        maze_size = self.maze_size  # - 2  # Without the wall
        
        maze = np.zeros([maze_size, maze_size]) 
        
        # List of all possible locations
        all_idx = np.array(list(cartesian_product(range(maze_size), range(maze_size))))

        # Randomly sample locations according to obstacle_ratio
        random_idx_idx = np.random.choice(maze_size**2, size=int(self.obstacle_ratio*maze_size**2), replace=False)
        random_obs_idx = all_idx[random_idx_idx]

        # Fill obstacles
        for idx in random_obs_idx:
            maze[idx[0], idx[1]] = 1

        # Padding with walls, i.e. ones
        maze = np.pad(maze, 1, 'constant', constant_values=1)
        
        return maze
