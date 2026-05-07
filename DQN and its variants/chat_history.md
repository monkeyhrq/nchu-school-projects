# AI 協作對話紀錄 - Homework 3: DQN and its Variants

## 協作目標
完成「深度強化學習」課程的 Homework 3，使用 `tf.keras` 與自定義 `GradientTape` 訓練迴圈，在 Gridworld 環境中循序漸進地實作 DQN 及其進階變體（包含 Double DQN、Dueling DQN 與 Prioritized Experience Replay）。所有實作皆以老師提供之 `Gridworld.py` 與 `GridBoard.py` 為環境基底，並對齊老師教材的神經網路架構（`64 → 150 → 100 → 4`）。

---

## 階段一：理解作業需求與環境

**學生**：這份作業是要做什麼？

**AI**：這份作業的主題是 Deep Q-Network（DQN）及其變體的實作。環境是一個 4×4 的 Gridworld，裡面有四種物件：Player、Goal、Pit、Wall。根據模式不同分為三種：
- **Static Mode**：所有物件位置固定，是最簡單的情況
- **Player Mode**：只有 Player 位置隨機，其他固定
- **Random Mode**：所有物件位置全部隨機，最困難

作業分三個部分：HW3-1 在 Static Mode 下跑基礎 DQN，HW3-2 在 Player Mode 下實作 Double DQN 與 Dueling DQN，HW3-3 用 Keras 改寫並在 Random Mode 下訓練，加入訓練技巧。

---

**學生**：狀態是怎麼表示的？

**AI**：Gridworld 的狀態用 `render_np()` 輸出，形狀是 `(4, 4, 4)`，也就是 4 個 channel 各代表一種物件（Player、Goal、Pit、Wall），每個 channel 是一張 4×4 的地圖，值為 0 或 1 表示該物件在不在那個格子。訓練時會把它攤平成 64 維的一維向量餵進神經網路。

---

**學生**：老師要我交什麼出去？

**AI**：根據作業說明，需要繳交：
1. **HW3-1（30%）**：跑 Naive DQN 與 Experience Replay Buffer，並繳交一份理解報告
2. **HW3-2（40%）**：實作 Double DQN 與 Dueling DQN，在 Player Mode 下比較它們如何改善基礎 DQN
3. **HW3-3（30%）**：把模型改寫成 Keras，在 Random Mode 下訓練，加入 Gradient Clipping、Learning Rate Scheduling 等技巧可以獲得加分

---

## 階段二：3-1 Static Mode 實作

**學生**：幫我完成 3-1。

**AI**：好，3-1 要實作的是基礎 DQN 加上 Experience Replay Buffer，在 Static Mode 下訓練。以老師提供的 `Gridworld.py` 為環境，網路架構對齊教材的 `64 → 150 → 100 → 4`，使用 Keras 搭配 `GradientTape` 自定義訓練迴圈。

主要元件：
- **ReplayBuffer**：用 `collections.deque` 實作，容量 1000，每次隨機抽 200 筆訓練
- **Epsilon-greedy**：從 1.0 線性衰減到 0.1，讓模型從探索逐漸轉向利用
- **Bellman equation**：`Y = reward + γ × max(Q(s'))`，γ = 0.9

---

**學生**：結果是什麼？

**AI**：Loss 曲線在訓練初期衝到約 3.7，隨後快速下降，大約 1000 steps 之後就趨近於 0 並穩定。這個結果很正常，代表模型在 Static Mode 下已經收斂，Agent 學會了從固定起點走到終點的最佳路徑。

---

**學生**：為什麼需要 Experience Replay？不加的話會怎樣？

**AI**：不加 Experience Replay 的話會發生「災難性遺忘（Catastrophic Forgetting）」。原因是這樣：

神經網路訓練需要資料盡量接近獨立同分布（i.i.d）的假設。但在 DQN 的訓練迴圈裡，連續產生的樣本（s1 → s2 → s3）是高度時間相關的，因為它們來自同一局遊戲的連續步驟。用這種資料直接更新網路，等於讓網路一直在「同一個方向」學習，會把之前學到的東西蓋掉。

Experience Replay 的做法是把每一步的經驗存進記憶池，訓練時從裡面隨機抽樣，打破了樣本之間的時間相關性，讓訓練資料的分布變得更均勻，訓練也更穩定。

---

**學生**：Static Mode 夠簡單所以不需要 Target Network 嗎？

**AI**：對，在 3-1 的 Static Mode 裡先不加 Target Network，因為環境夠簡單，狀態空間很小，用同一個網路計算 target Q 值也能穩定收斂。Target Network 主要是在環境更複雜、target 更不穩定的時候才會明顯發揮作用，所以留到 3-2 的 Player Mode 再加進去，比較能體現它的效果。

---

## 階段三：3-2 Player Mode 實作

**學生**：幫我繼續完成 3-2。

**AI**：3-2 要在 Player Mode 下實作 Double DQN 與 Dueling DQN，同時加入 Target Network。在 3-1 的基礎上主要改了兩個地方：

**網路架構改成 Dueling DQN**：共用特徵層之後分成兩條支流，Value Stream 輸出 V(s)，Advantage Stream 輸出 A(s,a)，最後合併為 `Q(s,a) = V(s) + A(s,a) - mean(A)`。

**Target Q 計算改成 Double DQN**：
```python
# 普通 DQN
Y = r + γ × max(Q_target(s'))

# Double DQN
best_action = argmax(Q_online(s'))      # online network 選動作
Y = r + γ × Q_target(s', best_action)  # target network 評估
```

---

**學生**：Double DQN 跟普通 DQN 差在哪？

**AI**：差在 target Q 值的計算方式。

普通 DQN 計算 target 的時候，用 target network 直接取 `max Q(s')`，這個 max 操作會系統性地高估 Q 值，因為它永遠選最大的那個，雜訊也會被放大。

Double DQN 把這個過程拆成兩步：先用 online network 選出它認為最好的動作，再用 target network 評估那個動作的 Q 值。選動作和評估是分開的，可以有效抑制 Q 值高估的問題，讓學習曲線更穩定。

---

**學生**：Dueling DQN 的優勢在哪裡？

**AI**：Dueling DQN 的核心想法是把 Q 值拆成兩個部分分開學：

- **V(s)**：這個狀態本身有多好，跟選哪個動作無關
- **A(s,a)**：在這個狀態下，選某個動作比平均好多少

在 Player Mode 這個設計特別有用，因為玩家起點是隨機的，不同格子的「本身價值」差很多（靠近 Goal 的格子就是比靠近 Pit 的格子好）。讓網路能直接學到這種狀態價值，不需要透過每個動作才能間接反映，收斂速度會快很多。

此外，在很多狀態下（比如撞牆或是離終點很遠），選哪個動作其實差不多爛，這種時候 Advantage 接近 0，網路就可以把精力集中在學 V(s)，不會浪費在無意義的動作區分上。

---

**學生**：結果怎樣？

**AI**：Loss 曲線在前 3000~4000 steps 有幾次明顯的波峰，這是 target network 同步時造成的短暫震盪，屬於正常現象。之後 Loss 快速下降並收斂到接近 0，整體訓練穩定。Player Mode 比 Static Mode 難，但模型仍然學會了從任意起點導航到終點的通用策略。

---

## 階段四：3-3 Random Mode 實作

**學生**：幫我繼續做 3-3。

**AI**：3-3 在 Random Mode 下訓練，這是最困難的模式，所有物件位置每回合都隨機生成。在 3-2 的基礎上加入了：

**S5：Prioritized Experience Replay (PER)**：不再均勻抽樣，而是根據每筆經驗的 TD Error 給予優先級，誤差越大被抽到的機率越高。同時用 Importance Sampling weights 修正非均勻抽樣造成的偏差。

**訓練技巧**：
- **Learning Rate Scheduling**：`ExponentialDecay`，初始 1e-3，每 1000 steps 衰減 0.9 倍
- **Gradient Clipping**：`clipnorm=1.0`，限制梯度最大值
- **Wall Collision Penalty**：撞牆給予 -5 懲罰，加速學習邊界規則

---

**學生**：為什麼 Random Mode 需要 PER？

**AI**：在 Random Mode 裡，每回合盤面都是全新的，找到 Goal（+10）或掉進 Pit（-10）的機率相對低，大部分經驗都是無意義的移動步驟（-1）。

如果用普通的均勻抽樣，這些稀有但高價值的經驗很快就會被大量普通步驟稀釋，被抽到的機率越來越低，等於模型幾乎學不到這些重要經驗。

PER 的做法是讓 TD Error 大的經驗有更高的機率被重複學習。TD Error 大代表「這個結果跟模型預期差很多」，也就是模型還沒學好的地方，讓它多練這些部分可以大幅提升樣本利用率。

---

**學生**：Gradient Clipping 是什麼？為什麼要用？

**AI**：Gradient Clipping 是在反向傳播更新參數之前，限制梯度向量的最大範數（norm）。在這裡設定 `clipnorm=1.0`，意思是如果梯度的 L2 norm 超過 1.0，就等比例縮小讓它剛好等於 1.0，方向不變但幅度被限制住。

在 Random Mode 裡需要用它的原因是：全隨機的盤面可能產生很大的 TD Error（比如模型完全沒預期到的盤面組合），這會讓梯度瞬間變得非常大，一次更新就把網路權重打壞，之前學到的東西全部毀掉。Gradient Clipping 就是防止這種「梯度爆炸」的保護機制。

---

**學生**：結果怎樣？

**AI**：Loss 從初期約 20 快速下降，中間有幾次小幅震盪，後期穩定收斂到接近 0。Random Mode 的訓練曲線比前兩個模式震盪幅度更大，這是環境本身複雜度造成的，屬於正常現象。PER 加入之後明顯讓後期的 Loss 更穩定，沒有出現大幅度的反彈。

---

## 階段五：整體比較與總結

**學生**：這三個模式加的東西不一樣，可以幫我整理一下每個機制是為了解決什麼問題嗎？

**AI**：

| 機制 | 解決的問題 | 從哪個階段加入 |
|---|---|---|
| S1 Experience Replay | 樣本時間相關性高，導致災難性遺忘 | 3-1 |
| S2 Target Network | Target Q 值不穩定，網路追著自己跑 | 3-2 |
| S3 Double DQN | Q 值系統性高估（Overestimation Bias） | 3-2 |
| S4 Dueling DQN | 無法有效學習狀態本身的價值 | 3-2 |
| S5 PER | 稀有高價值經驗被大量普通樣本稀釋 | 3-3 |
| Gradient Clipping | 隨機環境造成梯度爆炸 | 3-3 |
| LR Scheduling | 後期學習率過大導致收斂不穩定 | 3-3 |
| Wall Penalty | 模型學習邊界規則速度太慢 | 3-3 |

每個機制都是針對更難的環境所出現的新問題去加的，這也是為什麼作業要循序漸進從 Static → Player → Random 來做的原因。

---

**學生**：作業完成了嗎？

**AI**：三個階段均已完成，最終專案包含：
- `3_1_static.py`：Basic DQN + Experience Replay（Static Mode）
- `3_2_player.py`：Double DQN + Dueling DQN + Target Network（Player Mode）
- `3_3_random.py`：PER + Training Tips（Random Mode）
- `Gridworld.py` / `GridBoard.py`：老師提供的環境檔案
- `3_1_loss.png` / `3_2_loss.png` / `3_3_loss.png`：訓練結果圖
- `README.md`：完整分析報告
- `AI_協作對話紀錄.md`：本份協作紀錄

整份作業已達 100% 完成度。
