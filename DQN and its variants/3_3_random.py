import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
from Gridworld import Gridworld

# ==========================================
# 3-3: Enhanced DQN for Random Mode WITH Training Tips
# Mechanism S5: Prioritized Experience Replay
# Stabilization: Learning Rate Schedule, Gradient Clipping, Wall Penalty
# ==========================================

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
# 上下左右對應的位移
move_pos = [(-1,0),(1,0),(0,-1),(0,1)]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_prio
            
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        
        # IS Weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.concatenate(states, axis=0),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.concatenate(next_states, axis=0),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = err + 1e-5

    def __len__(self):
        return len(self.buffer)

def create_dueling_q_network():
    inputs = layers.Input(shape=(64,))
    x = layers.Dense(150, activation='relu')(inputs)
    x = layers.Dense(100, activation='relu')(x)
    value = layers.Dense(1, activation='linear')(x)
    advantage = layers.Dense(4, activation='linear')(x)
    advantage_mean = layers.Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
    outputs = layers.Add()([value, layers.Subtract()([advantage, advantage_mean])])
    return keras.Model(inputs=inputs, outputs=outputs)

def train():
    print("Starting 3-3: Random Mode Training (PER + Training Tips)...")
    
    gamma = 0.9
    epsilon = 1.0
    epochs = 2500
    batch_size = 200
    mem_size = 1000
    max_moves = 50
    sync_freq = 500
    
    q_network = create_dueling_q_network()
    target_network = create_dueling_q_network()
    target_network.set_weights(q_network.get_weights())
    
    # === Training Tips: LR Scheduling & Gradient Clipping ===
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    
    @tf.function
    def train_step(s1_tensor, s2_tensor, a_b, r_b, d_b, weights):
        Q1_next = q_network(s2_tensor)
        best_action_next = tf.argmax(Q1_next, axis=1)
        Q2_next = target_network(s2_tensor)
        action_masks_next = tf.one_hot(best_action_next, 4)
        Q_target_next = tf.reduce_sum(Q2_next * action_masks_next, axis=1)
        Y = r_b + gamma * ((1.0 - d_b) * Q_target_next)
        
        with tf.GradientTape() as tape:
            Q1 = q_network(s1_tensor)
            action_masks = tf.one_hot(a_b, 4)
            X = tf.reduce_sum(Q1 * action_masks, axis=1)
            td_errors = tf.abs(Y - X)
            loss = tf.reduce_mean(weights * tf.square(Y - X))
        
        grads = tape.gradient(loss, q_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
        return loss, td_errors

    replay = PrioritizedReplayBuffer(mem_size)
    losses = []
    step_count = 0
    beta = 0.4
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='random')
        state1_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        status = 1
        mov = 0
        
        while(status == 1):
            mov += 1
            step_count += 1
            state1_tensor = tf.convert_to_tensor(state1_, dtype=tf.float32)
            qval = q_network(state1_tensor)
            qval_ = qval.numpy()
            
            if (random.random() < epsilon):
                action_ = np.random.randint(0,4)
            else:
                action_ = np.argmax(qval_)
                
            # === Training Tips: Wall collision penalty ===
            hit_wall = game.validateMove('Player', move_pos[action_]) == 1
            
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
            
            # 若撞牆給予額外懲罰 -5，加速學會避開牆壁
            reward = -5 if hit_wall else game.reward()
            
            done = True if reward != -1 and not hit_wall else False 
            # 這裡必須注意，只有真正抵達終點或陷阱才算 done，撞牆不算 done
            if abs(reward) == 10:
                done = True
            
            replay.add(state1_, action_, reward, state2_, done)
            state1_ = state2_
            
            if len(replay) > batch_size:
                s1_b, a_b, r_b, s2_b, d_b, indices, weights = replay.sample(batch_size, beta=beta)
                s1_tensor = tf.convert_to_tensor(s1_b, dtype=tf.float32)
                s2_tensor = tf.convert_to_tensor(s2_b, dtype=tf.float32)
                
                loss, td_errors = train_step(s1_tensor, s2_tensor, tf.convert_to_tensor(a_b, dtype=tf.int32), tf.convert_to_tensor(r_b, dtype=tf.float32), tf.convert_to_tensor(d_b, dtype=tf.float32), tf.convert_to_tensor(weights, dtype=tf.float32))
                
                # S5: Update priorities in PER
                replay.update_priorities(indices, td_errors.numpy())
                losses.append(loss.numpy())
            
            if step_count % sync_freq == 0:
                target_network.set_weights(q_network.get_weights())
                
            if abs(reward) == 10 or mov > max_moves:
                status = 0
                mov = 0
                
        if epsilon > 0.1:
            epsilon -= (1/epochs)
        
        beta = min(1.0, beta + 0.001)
            
        if (i+1) % 100 == 0:
            avg_loss = np.mean(losses[-100:]) if len(losses) > 0 else 0
            print(f"Epoch {i+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.2f}")

    plt.figure(figsize=(10,7))
    plt.plot(losses)
    plt.xlabel("Steps", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("3-3: Random Mode (PER + Training Tips)")
    plt.savefig('3_3_loss.png')
    print("Training finished. Loss plot saved to 3_3_loss.png")

if __name__ == "__main__":
    train()

"""
# 分析與報告 (3-3)

**環境難度分析 (Random Mode)**:
極端困難。Player、Goal、Pit、Wall 的位置在每一回合都會完全隨機生成。盤面的狀態空間呈現爆炸性增長，模型必須真正學會物件之間的「相對動態空間關係」，而不能只靠背座標。

**訓練不穩定症狀 (若不使用 PER 與 Training Tips)**:
在全隨機環境中，找到目標 (+10) 或是掉進陷阱 (-10) 的經驗變得相對稀少，大部分的經驗都是在無意義地移動 (-1)。若使用一般均勻抽樣的 Replay Buffer，模型會被大量無用經驗稀釋，學習極度緩慢甚至卡在區域最佳解。同時，由於每次盤面差異極大，TD Target 計算會產生巨大的梯度變異，導致 Loss 劇烈震盪，破壞原本學好的權重。

**選擇的 DQN 機制 (S5: PER) 與 Training Tips**:
- **S5 Prioritized Experience Replay (PER)**: 為每次經驗計算 TD Error，誤差越大的經驗（代表模型越驚訝、沒學好的部分，如意外抵達終點）被抽樣重放的機率越高。這解決了稀有經驗被稀釋的問題，極大提升了樣本利用率。
- **Learning Rate Schedule**: 初期給予較大的學習率以便在龐大隨機空間中快速探索，後期指數衰減學習率，穩定網路收斂。
- **Gradient Clipping (`clipnorm=1.0`)**: 由於隨機環境造成的 TD Error 變異極大，限制梯度的最大值能保護網路權重不崩潰。
- **Wall Collision Penalty (Reward Shaping)**: 參考老師的進階技巧，給予撞牆 -5 的懲罰，強迫模型更快理解邊界規則，加速訓練速度。
"""
