import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
from collections import deque
from Gridworld import Gridworld

# ==========================================
# 3-2: Enhanced DQN Variants for Player Mode
# Mechanism S2: Target Network
# Mechanism S3: Double DQN
# Mechanism S4: Dueling DQN
# ==========================================

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.concatenate(states, axis=0),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.concatenate(next_states, axis=0),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

def create_dueling_q_network():
    # S4: Dueling DQN Architecture
    inputs = layers.Input(shape=(64,))
    x = layers.Dense(150, activation='relu')(inputs)
    x = layers.Dense(100, activation='relu')(x)
    
    # State-Value Stream V(s)
    value = layers.Dense(1, activation='linear')(x)
    
    # Advantage Stream A(s, a)
    advantage = layers.Dense(4, activation='linear')(x)
    
    # Combine V(s) and A(s, a) -> Q(s, a)
    advantage_mean = layers.Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
    outputs = layers.Add()([value, layers.Subtract()([advantage, advantage_mean])])
    
    return keras.Model(inputs=inputs, outputs=outputs)

def train():
    print("Starting 3-2: Player Mode Training (Double & Dueling DQN)...")
    
    # Hyperparameters
    gamma = 0.9
    epsilon = 1.0
    epochs = 2000
    batch_size = 200
    mem_size = 1000
    learning_rate = 1e-3
    max_moves = 50
    sync_freq = 500 # Target Network 同步頻率
    
    # Primary network and Target network (S2)
    q_network = create_dueling_q_network()
    target_network = create_dueling_q_network()
    target_network.set_weights(q_network.get_weights())
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.MeanSquaredError()
    replay = ReplayBuffer(mem_size)
    
    @tf.function
    def train_step(s1_tensor, s2_tensor, a_b, r_b, d_b):
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
            loss = loss_fn(Y, X)
        
        grads = tape.gradient(loss, q_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
        return loss

    losses = []
    step_count = 0
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='player') # Player Mode
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
                
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
            reward = game.reward()
            
            done = True if reward != -1 else False
            replay.add(state1_, action_, reward, state2_, done)
            state1_ = state2_
            
            if len(replay) > batch_size:
                s1_b, a_b, r_b, s2_b, d_b = replay.sample(batch_size)
                s1_tensor = tf.convert_to_tensor(s1_b, dtype=tf.float32)
                s2_tensor = tf.convert_to_tensor(s2_b, dtype=tf.float32)
                
                loss = train_step(s1_tensor, s2_tensor, tf.convert_to_tensor(a_b, dtype=tf.int32), tf.convert_to_tensor(r_b, dtype=tf.float32), tf.convert_to_tensor(d_b, dtype=tf.float32))
                losses.append(loss.numpy())
            
            # S2: Sync Target Network
            if step_count % sync_freq == 0:
                target_network.set_weights(q_network.get_weights())
                
            if abs(reward) == 10 or mov > max_moves:
                status = 0
                mov = 0
                
        if epsilon > 0.1:
            epsilon -= (1/epochs)
            
        if (i+1) % 100 == 0:
            avg_loss = np.mean(losses[-100:]) if len(losses) > 0 else 0
            print(f"Epoch {i+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.2f}")

    plt.figure(figsize=(10,7))
    plt.plot(losses)
    plt.xlabel("Steps", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("3-2: Player Mode (Double + Dueling DQN)")
    plt.savefig('3_2_loss.png')
    print("Training finished. Loss plot saved to 3_2_loss.png")

if __name__ == "__main__":
    train()

"""
# 分析與報告 (3-2)

**環境難度分析 (Player Mode)**:
中等。Player 的出生點隨機，其他物件固定。Agent 無法只依賴死記單一軌跡，必須學習到整個空間的「泛化策略」，知道在任何位置該往哪個方向走。

**訓練不穩定症狀 (若不使用 Double/Dueling)**:
在起點隨機的情況下，若只使用 Basic DQN，很容易發生「Q值高估 (Overestimation Bias)」，因為 `max` 運算會將雜訊放大。這會導致模型變得過度樂觀，無法收斂。同時，在離終點很遠的格子，或是撞牆的格子，由於價值極低，若使用單一流 (Single Stream) 的網路，學習效率會很差，因為網路需要為每個獨立的動作分別學會「在這個狀態很爛」。

**選擇的 DQN 機制 (S3: Double DQN, S4: Dueling DQN)**:
- **S3 Double DQN**: 將「選擇動作」與「評估動作價值」的神經網路拆開（利用 Target Network 評估 Primary Network 選出來的動作）。這極大程度地抑制了 Q 值的爆炸與高估。
- **S4 Dueling DQN**: 將網路拆分為「State-Value (V)」與「Advantage (A)」。這對於 Player Mode 非常關鍵，因為模型可以直接評估「目前這個格子有多好 (V)」，而不必管接下來要走哪裡。這大幅加速了從不同隨機起點出發時的價值收斂。
"""
