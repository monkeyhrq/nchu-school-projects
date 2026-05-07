import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
from collections import deque
from gridworld import Gridworld
import os

# ==========================================
# Stage 1: Naive DQN for Static Mode
# Mechanism S1: Replay Buffer
# Mechanism S2: Target Network
# ==========================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

def create_q_network(state_shape, action_dim):
    inputs = layers.Input(shape=state_shape)
    # Flatten the gridworld state
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(action_dim, activation='linear')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

def train_stage1():
    print("Starting Stage 1: Static Mode Training...")
    env = Gridworld(mode='static')
    
    # Hyperparameters
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.99
    batch_size = 32
    buffer_capacity = 1000
    learning_rate = 1e-3
    update_target_freq = 50
    episodes = 200
    max_steps = 50

    q_network = create_q_network(env.state_shape, env.action_space_n)
    target_network = create_q_network(env.state_shape, env.action_space_n)
    target_network.set_weights(q_network.get_weights())
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.MeanSquaredError()
    buffer = ReplayBuffer(buffer_capacity)
    
    reward_history = []
    loss_history = []
    step_count = 0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(0, env.action_space_n)
            else:
                state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
                q_values = q_network(state_tensor)
                action = tf.argmax(q_values[0]).numpy()
                
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Training Step
            if len(buffer) >= batch_size:
                b_states, b_actions, b_rewards, b_next_states, b_dones = buffer.sample(batch_size)
                
                # S2: Target Network computation
                next_q_values = target_network(b_next_states)
                max_next_q = tf.reduce_max(next_q_values, axis=1)
                td_targets = b_rewards + (1.0 - b_dones) * gamma * max_next_q
                
                with tf.GradientTape() as tape:
                    # Q-values for current states
                    q_values = q_network(b_states)
                    
                    # Select Q-values for chosen actions
                    action_masks = tf.one_hot(b_actions, env.action_space_n)
                    q_action = tf.reduce_sum(q_values * action_masks, axis=1)
                    
                    # Compute loss
                    loss = loss_fn(td_targets, q_action)
                
                grads = tape.gradient(loss, q_network.trainable_variables)
                optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
                episode_loss.append(loss.numpy())
            
            # S2: Target Network Sync
            if step_count % update_target_freq == 0:
                target_network.set_weights(q_network.get_weights())
                
            if done:
                break
                
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reward_history.append(episode_reward)
        if episode_loss:
            loss_history.append(np.mean(episode_loss))
        else:
            loss_history.append(0)
            
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1}/{episodes} | Reward: {episode_reward} | Epsilon: {epsilon:.2f}")

    print("Training Finished.")
    
    # Save plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(reward_history)
    plt.title('Stage 1: Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss_history)
    plt.title('Stage 1: Loss over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    
    plt.tight_layout()
    plt.savefig('stage1_results.png')
    print("Saved training results to stage1_results.png")

if __name__ == "__main__":
    train_stage1()

"""
# Stage 1 Analysis

**Environment Difficulty**: Very Easy (Static Mode). 
All objects are always in the same place. The agent only needs to memorize a single optimal trajectory from (0,3) to (0,0).

**Training Instability Symptoms without S1/S2**:
If we don't use a Replay Buffer (S1) or Target Network (S2), the agent updates its weights on highly correlated, sequential states. This causes "Catastrophic Forgetting" and erratic loss spikes. The TD target would be constantly shifting because the network generating the target is the same one being updated.

**Which DQN Weakness Appears**:
Sample correlation and non-stationary targets.

**Why the selected scheme solves the problem**:
- **S1 Replay Buffer**: Randomly samples past experiences, breaking the temporal correlation between consecutive states. It stabilizes the input distribution.
- **S2 Target Network**: Freezes the Q-network used to compute TD targets for a set number of steps (`update_target_freq`). This provides a stationary target for the loss function, preventing the network from chasing its own tail.

**Why other schemes are skipped**:
Double DQN, Dueling DQN, and PER are overkill for a completely deterministic and static environment. The state space is so small and fixed that basic DQN converges reliably and extremely fast.
"""
