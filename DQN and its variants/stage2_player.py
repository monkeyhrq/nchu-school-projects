import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from collections import deque
import random
from gridworld import Gridworld

# ==========================================
# Stage 2: Enhanced DQN for Player Mode
# Mechanism S3: Double DQN
# Mechanism S4: Dueling DQN
# Included: S1 (Replay Buffer), S2 (Target Network)
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

def create_dueling_q_network(state_shape, action_dim):
    """
    S4: Dueling DQN Architecture
    Separates the network into a Value stream and an Advantage stream.
    """
    inputs = layers.Input(shape=state_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Value Stream
    value = layers.Dense(64, activation='relu')(x)
    value = layers.Dense(1, activation='linear')(value)
    
    # Advantage Stream
    advantage = layers.Dense(64, activation='relu')(x)
    advantage = layers.Dense(action_dim, activation='linear')(advantage)
    
    # Combine Value and Advantage
    # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    advantage_mean = keras.ops.mean(advantage, axis=1, keepdims=True)
    q_values = value + (advantage - advantage_mean)
    
    return keras.Model(inputs=inputs, outputs=q_values)

def train_stage2():
    print("Starting Stage 2: Player Mode Training (Double & Dueling DQN)...")
    env = Gridworld(mode='player')
    
    # Hyperparameters
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    batch_size = 64
    buffer_capacity = 5000
    learning_rate = 5e-4
    update_target_freq = 100
    episodes = 500
    max_steps = 50

    q_network = create_dueling_q_network(env.state_shape, env.action_space_n)
    target_network = create_dueling_q_network(env.state_shape, env.action_space_n)
    target_network.set_weights(q_network.get_weights())
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.Huber() # Huber loss is more robust to outliers
    buffer = ReplayBuffer(buffer_capacity)
    
    reward_history = []
    loss_history = []
    step_count = 0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
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
            
            if len(buffer) >= batch_size:
                b_states, b_actions, b_rewards, b_next_states, b_dones = buffer.sample(batch_size)
                
                # S3: Double DQN computation
                # 1. Use primary network to select best action
                next_q_values_primary = q_network(b_next_states)
                best_next_actions = tf.argmax(next_q_values_primary, axis=1)
                
                # 2. Use target network to evaluate the chosen action
                next_q_values_target = target_network(b_next_states)
                action_masks_next = tf.one_hot(best_next_actions, env.action_space_n)
                target_q_values = tf.reduce_sum(next_q_values_target * action_masks_next, axis=1)
                
                td_targets = b_rewards + (1.0 - b_dones) * gamma * target_q_values
                
                with tf.GradientTape() as tape:
                    q_values = q_network(b_states)
                    action_masks = tf.one_hot(b_actions, env.action_space_n)
                    q_action = tf.reduce_sum(q_values * action_masks, axis=1)
                    loss = loss_fn(td_targets, q_action)
                
                grads = tape.gradient(loss, q_network.trainable_variables)
                optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
                episode_loss.append(loss.numpy())
            
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
            
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{episodes} | Avg Reward (last 50): {np.mean(reward_history[-50:]):.2f} | Epsilon: {epsilon:.2f}")

    print("Training Finished.")
    
    # Save plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(reward_history)
    plt.title('Stage 2: Rewards (Double + Dueling)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss_history)
    plt.title('Stage 2: Loss (Double + Dueling)')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    
    plt.tight_layout()
    plt.savefig('stage2_results.png')
    print("Saved training results to stage2_results.png")

if __name__ == "__main__":
    train_stage2()

"""
# Stage 2 Analysis

**Environment Difficulty**: Moderate (Player Mode). 
The player's starting position is randomized. The agent must generalize its learning to know how to navigate from any starting block to the goal without hitting the pit or wall.

**Training Instability Symptoms without S3/S4**:
With a standard DQN in a stochastic starting environment, the max operation in computing the TD target `max(Q(s', a'))` systematically overestimates the true Q-values. If Q-values explode, the policy becomes sub-optimal, and loss spikes. Additionally, evaluating `Q(s,a)` when the action `a` has little impact (e.g., hitting a wall, or far away from goal) is slow to learn using a single stream.

**Which DQN Weakness Appears**:
1. **Overestimation Bias**: Standard DQN uses the same network to select an action and evaluate it, leading to optimistic value estimates.
2. **Redundant Action Evaluation**: In many states, the value of the state itself is what matters (e.g., being next to the goal is good, regardless of whether you move UP or LEFT, if both reach it).

**Why the selected scheme solves the problem**:
- **S3 Double DQN**: Decouples action selection from action evaluation. We use the *primary* network to select the best action in the next state, and the *target* network to evaluate its Q-value. This prevents over-optimistic Q-value explosions.
- **S4 Dueling DQN**: Splits the network into two streams: `V(s)` (State-Value) and `A(s, a)` (Advantage). This helps the network learn which states are inherently valuable, independent of the action taken. This is incredibly useful here because the Player position constantly changes. Knowing the underlying value of a coordinate speeds up learning tremendously.

**Why other schemes are skipped**:
PER is not yet strictly necessary. Random player start makes the environment harder, but the state space is still bounded to 16 positions, and standard uniform sampling finds the goal frequently enough.
"""
