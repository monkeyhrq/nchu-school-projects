import numpy as np

class Gridworld:
    def __init__(self, mode='static'):
        """
        mode: 'static', 'player', 'random'
        Grid is 4x4.
        Objects: 0: Player, 1: Goal, 2: Pit, 3: Wall
        Rewards: Goal +10, Pit -10, step -1.
        """
        self.mode = mode
        self.size = 4
        self.state_shape = (self.size, self.size, 4)
        self.action_space_n = 4 # 0: Up, 1: Down, 2: Left, 3: Right
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size, 4), dtype=np.float32)
        
        # Determine positions based on mode
        if self.mode == 'static':
            self.player_pos = (0, 3) # row, col (Wait, the assignment image says Player at (0,3). Assuming 0-indexed: row 0, col 3. Or x,y? Usually y,x) Let's assume standard row, col.
            self.goal_pos = (0, 0)
            self.pit_pos = (0, 1)
            self.wall_pos = (1, 1)
        elif self.mode == 'player':
            self.goal_pos = (0, 0)
            self.pit_pos = (0, 1)
            self.wall_pos = (1, 1)
            self.player_pos = self._get_random_empty_pos([self.goal_pos, self.pit_pos, self.wall_pos])
        elif self.mode == 'random':
            self.goal_pos = self._get_random_empty_pos([])
            self.pit_pos = self._get_random_empty_pos([self.goal_pos])
            self.wall_pos = self._get_random_empty_pos([self.goal_pos, self.pit_pos])
            self.player_pos = self._get_random_empty_pos([self.goal_pos, self.pit_pos, self.wall_pos])
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self._update_board()
        return self.board.copy()

    def _get_random_empty_pos(self, occupied):
        while True:
            r, c = np.random.randint(0, self.size), np.random.randint(0, self.size)
            if (r, c) not in occupied:
                return (r, c)

    def _update_board(self):
        self.board.fill(0)
        self.board[self.player_pos[0], self.player_pos[1], 0] = 1
        self.board[self.goal_pos[0], self.goal_pos[1], 1] = 1
        self.board[self.pit_pos[0], self.pit_pos[1], 2] = 1
        self.board[self.wall_pos[0], self.wall_pos[1], 3] = 1

    def step(self, action):
        """
        action: 0: Up, 1: Down, 2: Left, 3: Right
        """
        r, c = self.player_pos
        
        if action == 0:   # Up
            r = max(0, r - 1)
        elif action == 1: # Down
            r = min(self.size - 1, r + 1)
        elif action == 2: # Left
            c = max(0, c - 1)
        elif action == 3: # Right
            c = min(self.size - 1, c + 1)

        # Check wall collision
        if (r, c) == self.wall_pos:
            r, c = self.player_pos # Cannot move into wall

        self.player_pos = (r, c)
        self._update_board()

        done = False
        reward = -1 # step penalty

        if self.player_pos == self.goal_pos:
            reward = 10
            done = True
        elif self.player_pos == self.pit_pos:
            reward = -10
            done = True

        return self.board.copy(), reward, done, {}
