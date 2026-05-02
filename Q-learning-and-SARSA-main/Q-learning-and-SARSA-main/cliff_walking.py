import numpy as np

class CliffWalkingEnv:
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start_state = (3, 0)
        self.goal_state = (3, 11)
        self.state = self.start_state
        self.action_space = [0, 1, 2, 3] # UP, RIGHT, DOWN, LEFT
        
    def reset(self):
        self.state = self.start_state
        return self.state
        
    def step(self, action):
        r, c = self.state
        
        # Calculate next position
        if action == 0:   # UP
            r = max(0, r - 1)
        elif action == 1: # RIGHT
            c = min(self.cols - 1, c + 1)
        elif action == 2: # DOWN
            r = min(self.rows - 1, r + 1)
        elif action == 3: # LEFT
            c = max(0, c - 1)
            
        next_state = (r, c)
        
        # Check if step into cliff
        if r == 3 and 1 <= c <= 10:
            reward = -100
            next_state = self.start_state
            done = False
        # Check if reached goal
        elif next_state == self.goal_state:
            reward = -1
            done = True
        else:
            reward = -1
            done = False
            
        self.state = next_state
        return next_state, reward, done

    def __str__(self):
        # A simple visualization function to help debugging
        grid = np.zeros((self.rows, self.cols), dtype=str)
        grid[:] = '.'
        grid[3, 1:11] = 'C' # Cliff
        grid[3, 0] = 'S'    # Start
        grid[3, 11] = 'G'   # Goal
        grid[self.state[0], self.state[1]] = 'A' # Agent
        return '\n'.join([' '.join(row) for row in grid])
