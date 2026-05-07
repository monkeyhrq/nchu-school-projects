import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from collections import deque
import random
from gridworld import Gridworld

# ==========================================
# Stage 3: Enhanced DQN for Random Mode
# Mechanism S5: Prioritized Experience Replay
# Stabilization Tricks: LR Scheduling, Gradient Clipping
# Included: S1, S2, S3, S4
# ==========================================

class PrioritizedReplayBuffer:
    """
    S5: Prioritized Experience Replay (Simplified Array-based)
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-5

    def add(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_prio
            
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        
        # Importance Sampling weights
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max() # Normalize
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = err + self.epsilon

    def __len__(self):
        return len(self.buffer)

def create_dueling_q_network(state_shape, action_dim):
    inputs = layers.Input(shape=state_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    
    value = layers.Dense(64, activation='relu')(x)
    value = layers.Dense(1, activation='linear')(value)
    
    advantage = layers.Dense(64, activation='relu')(x)
    advantage = layers.Dense(action_dim, activation='linear')(advantage)
    
    advantage_mean = keras.ops.mean(advantage, axis=1, keepdims=True)
    q_values = value + (advantage - advantage_mean)
    
    return keras.Model(inputs=inputs, outputs=q_values)

def train_stage3():
    print("Starting Stage 3: Random Mode Training (PER + Tricks)...")
    env = Gridworld(mode='random')
    
    # Hyperparameters
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    batch_size = 64
    buffer_capacity = 10000
    update_target_freq = 200
    episodes = 800
    max_steps = 50

    q_network = create_dueling_q_network(env.state_shape, env.action_space_n)
    target_network = create_dueling_q_network(env.state_shape, env.action_space_n)
    target_network.set_weights(q_network.get_weights())
    
    # Stabilization trick 1: Learning Rate Scheduling
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=5000,
        decay_rate=0.9
    )
    
    # Stabilization trick 2: Gradient Clipping (clipnorm)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    
    buffer = PrioritizedReplayBuffer(buffer_capacity)
    
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
                b_states, b_actions, b_rewards, b_next_states, b_dones, indices, weights = buffer.sample(batch_size)
                
                next_q_values_primary = q_network(b_next_states)
                best_next_actions = tf.argmax(next_q_values_primary, axis=1)
                
                next_q_values_target = target_network(b_next_states)
                action_masks_next = tf.one_hot(best_next_actions, env.action_space_n)
                target_q_values = tf.reduce_sum(next_q_values_target * action_masks_next, axis=1)
                
                td_targets = b_rewards + (1.0 - b_dones) * gamma * target_q_values
                
                with tf.GradientTape() as tape:
                    q_values = q_network(b_states)
                    action_masks = tf.one_hot(b_actions, env.action_space_n)
                    q_action = tf.reduce_sum(q_values * action_masks, axis=1)
                    
                    # Compute TD Error for PER update
                    td_errors = tf.abs(td_targets - q_action)
                    
                    # Weighted MSE Loss
                    loss = tf.reduce_mean(weights * tf.square(td_targets - q_action))
                
                grads = tape.gradient(loss, q_network.trainable_variables)
                optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
                
                buffer.update_priorities(indices, td_errors.numpy())
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
            current_lr = float(keras.ops.convert_to_numpy(optimizer.learning_rate)) if hasattr(optimizer.learning_rate, 'numpy') else float(optimizer.learning_rate)
            print(f"Episode {episode+1}/{episodes} | Avg Reward (last 50): {np.mean(reward_history[-50:]):.2f} | Epsilon: {epsilon:.2f} | LR: {current_lr:.5f}")

    print("Training Finished.")
    
    # Save plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(reward_history)
    plt.title('Stage 3: Rewards (Random Mode + PER)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss_history)
    plt.title('Stage 3: Loss (Random Mode + PER)')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    
    plt.tight_layout()
    plt.savefig('stage3_results.png')
    print("Saved training results to stage3_results.png")

if __name__ == "__main__":
    train_stage3()

"""
# Stage 3 Analysis

**Environment Difficulty**: Hard (Random Mode). 
The player, goal, pit, and wall all spawn randomly. The state space jumps massively. The agent must understand the spatial relationship between all entities dynamically.

**Training Instability Symptoms without S5/Tricks**:
Uniform sampling is inefficient. Because the goal moves constantly, successful trajectories (hitting the goal) are rare and quickly buried by thousands of useless "wandering" steps in the replay buffer. Loss may plateau early, and the agent fails to converge on a generalized policy, exhibiting chaotic learning curves and failing to adapt to new random layouts.

**Which DQN Weakness Appears**:
1. **Sample Inefficiency**: Standard replay treats all experiences equally. Informative transitions (like finding the goal or falling in a pit) are sampled as often as meaningless steps into walls.
2. **Gradient Explosion**: The large state space and shifting target causes severe variance in TD errors, which can cause large gradients that break the network weights.

**Why the selected scheme solves the problem**:
- **S5 Prioritized Experience Replay (PER)**: Computes the TD-error for each transition. Experiences with higher TD-errors (meaning they surprised the network) are sampled more frequently. This forces the network to learn immediately from its biggest mistakes and most rewarding discoveries. We also use Importance Sampling weights to correct the bias introduced by non-uniform sampling.
- **Stabilization Tricks**:
  - **Learning Rate Scheduling**: Slowly decays the learning rate, allowing aggressive exploration early on and fine-tuning convergence later.
  - **Gradient Clipping (`clipnorm=1.0`)**: Caps the maximum gradient step size. If a huge TD error occurs, the network won't update its weights so drastically that it "forgets" everything else, ensuring stability.

**Why other schemes are skipped**:
No further core DQN variants are necessary (like NoisyNets or N-step) since PER + Dueling + Double DQN generally solves this environment scale. The environment is now fully robust.
"""
