import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
from collections import deque
from Gridworld import Gridworld

# ==========================================
# 3-1: Naive DQN & Experience Replay for Static Mode
# Mechanism S1: Replay Buffer
# ==========================================

action_set = {
    0: 'u', # Up
    1: 'd', # Down
    2: 'l', # Left
    3: 'r'  # Right
}

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

def create_q_network():
    # 對齊老師教材的架構 L1=64, L2=150, L3=100, L4=4
    inputs = layers.Input(shape=(64,))
    x = layers.Dense(150, activation='relu')(inputs)
    x = layers.Dense(100, activation='relu')(x)
    outputs = layers.Dense(4, activation='linear')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

def train():
    print("Starting 3-1: Static Mode Training (Basic DQN + Replay Buffer)...")
    
    # 超參數設定
    gamma = 0.9
    epsilon = 1.0
    epochs = 1500
    batch_size = 200
    mem_size = 1000
    learning_rate = 1e-3
    max_moves = 50
    
    q_network = create_q_network()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.MeanSquaredError()
    replay = ReplayBuffer(mem_size)
    
    @tf.function
    def train_step(s1_tensor, s2_tensor, a_b, r_b, d_b):
        with tf.GradientTape() as tape:
            Q1 = q_network(s1_tensor)
            Q2 = q_network(s2_tensor)
            maxQ = tf.reduce_max(Q2, axis=1)
            Y = r_b + gamma * ((1.0 - d_b) * maxQ)
            action_masks = tf.one_hot(a_b, 4)
            X = tf.reduce_sum(Q1 * action_masks, axis=1)
            loss = loss_fn(Y, X)
        grads = tape.gradient(loss, q_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
        return loss

    losses = []
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='static')
        state1_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        status = 1
        mov = 0
        
        while(status == 1):
            mov += 1
            # 轉換為 Tensor 以供網路輸入
            state1_tensor = tf.convert_to_tensor(state1_, dtype=tf.float32)
            qval = q_network(state1_tensor)
            qval_ = qval.numpy()
            
            # Epsilon-greedy
            if (random.random() < epsilon):
                action_ = np.random.randint(0,4)
            else:
                action_ = np.argmax(qval_)
                
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
            reward = game.reward()
            
            # 判斷是否結束 (到達目標+10 或 掉進陷阱-10)
            done = True if reward != -1 else False
            replay.add(state1_, action_, reward, state2_, done)
            state1_ = state2_
            
            # 經驗回放 (S1 機制)
            if len(replay) > batch_size:
                s1_b, a_b, r_b, s2_b, d_b = replay.sample(batch_size)
                s1_tensor = tf.convert_to_tensor(s1_b, dtype=tf.float32)
                s2_tensor = tf.convert_to_tensor(s2_b, dtype=tf.float32)
                
                loss = train_step(s1_tensor, s2_tensor, tf.convert_to_tensor(a_b, dtype=tf.int32), tf.convert_to_tensor(r_b, dtype=tf.float32), tf.convert_to_tensor(d_b, dtype=tf.float32))
                losses.append(loss.numpy())
                
            if abs(reward) == 10 or mov > max_moves:
                status = 0
                mov = 0
                
        if epsilon > 0.1:
            epsilon -= (1/epochs)
            
        if (i+1) % 100 == 0:
            avg_loss = np.mean(losses[-100:]) if len(losses) > 0 else 0
            print(f"Epoch {i+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.2f}")

    # 繪製並儲存 Loss 圖
    plt.figure(figsize=(10,7))
    plt.plot(losses)
    plt.xlabel("Steps", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("3-1: Static Mode (Basic DQN + Replay Buffer)")
    plt.savefig('3_1_loss.png')
    print("Training finished. Loss plot saved to 3_1_loss.png")

if __name__ == "__main__":
    train()

"""
# 分析與報告 (3-1)

**環境難度分析 (Static Mode)**:
極低。所有物件（玩家、目標、陷阱、牆壁）位置均固定不變。Agent 只需要找到並記住一條從起點到終點的固定路徑。

**訓練不穩定症狀 (若不使用 Experience Replay)**:
若只使用最原始的 Naive DQN (即老師教材一開始未加上 deque 的版本)，會發現 Loss 呈現極不穩定的震盪，發生「災難性遺忘 (Catastrophic Forgetting)」。這是因為連續收集到的樣本 (s, a, r, s') 是高度時間相關的，違反了類神經網路訓練時需要的 i.i.d (獨立同分佈) 假設。

**選擇的 DQN 機制 (S1: Replay Buffer)**:
使用 `collections.deque` 建立經驗回放池 (Replay Buffer)。
我們將每一步的經驗存入 buffer 中，然後隨機抽樣 (batch_size=200) 來進行訓練。這打破了樣本之間的時間相關性，平滑了資料分佈，讓神經網路可以穩定學習，有效降低 Loss 並收斂。在此簡單的靜態環境中，S1 足以讓模型學會最佳策略。
"""
