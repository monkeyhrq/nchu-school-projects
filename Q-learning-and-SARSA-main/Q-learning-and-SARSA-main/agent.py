import numpy as np
import random

class RLAgent:
    def __init__(self, rows, cols, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.rows = rows
        self.cols = cols
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Initialize Q-table: (rows, cols, num_actions)
        self.q_table = np.zeros((rows, cols, len(actions)))
        
    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            r, c = state
            q_values = self.q_table[r, c]
            # Handle ties randomly instead of always picking the first action
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return random.choice(best_actions)

class QLearningAgent(RLAgent):
    def update(self, state, action, reward, next_state, next_action, done):
        """
        Q-learning update (off-policy).
        """
        r, c = state
        nr, nc = next_state
        
        # In Q-learning, the target uses the max Q value of the next state
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[nr, nc])
            
        # Update rule
        current_q = self.q_table[r, c, action]
        self.q_table[r, c, action] = current_q + self.alpha * (target - current_q)

class SarsaAgent(RLAgent):
    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update (on-policy).
        """
        r, c = state
        nr, nc = next_state
        
        # In SARSA, the target uses the Q value of the next action actually chosen
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[nr, nc, next_action]
            
        # Update rule
        current_q = self.q_table[r, c, action]
        self.q_table[r, c, action] = current_q + self.alpha * (target - current_q)
