# 強化學習：Grid World 與馬可夫決策過程 (MDP)

🚀 **線上即時展示 (Live Demo)：[https://gridworld-nx4g.onrender.com/](https://gridworld-nx4g.onrender.com/)**

本專案實作了強化學習 (Reinforcement Learning) 中經典的「網格世界 (Grid World)」環境。
採用 **「前後端分離」** 的標準 Web 架構：以 **Flask (Python)** 作為後端處理核心演算法，並以 **HTML/CSS/JavaScript** 構建前端互動式無縫網格介面。使用者可以在同一個網頁中完成馬可夫決策過程 (MDP) 的三個核心階段。

## 🧠 核心功能與演算法實作

### 1. HW1-1: 網格地圖開發 (Grid Map Development)
* **前端互動介面**：動態生成大小為 $n \times n$ 的互動式網格（支援 3~10 維度）。
* **狀態標記**：透過 JavaScript 監聽單元格點擊事件，依序設定：
  * **起點 (Start)**：顯示為綠色 (S)
  * **終點 (End)**：顯示為紅色 (E)
  * **障礙物 (Obstacle)**：顯示為灰色 (X)

### 2. HW1-2: 策略顯示與價值評估 (Policy & Value Evaluation)
* **均勻隨機策略 (Uniform Random Policy)**：在每個非終點、非障礙物的狀態下，向上下左右四個方向移動的機率均等 ($\pi(a|s) = 0.25$)。
* **策略評估 (Policy Evaluation)**：後端透過 AJAX 接收網格狀態後，使用**貝爾曼期望方程式 (Bellman Expectation Equation)** 進行迭代計算：
  $$V_{k+1}(s) = \sum_{a} \pi(a|s) \left( \mathcal{R}_s^a + \gamma V_k(s') \right)$$
* **視覺化**：將收斂後的對稱價值矩陣 $V(s)$ 與隨機動作箭頭，透過 JSON 傳回前端並動態渲染至表格中。

### 3. HW1-3: 價值迭代算法推導最佳政策 (Optimal Policy via Value Iteration)
* **價值迭代 (Value Iteration)**：為了找出抵達終點的最佳路徑，演算法改用**貝爾曼最佳方程式 (Bellman Optimality Equation)**，每次迭代皆選擇能帶來最大期望回報的行動：
  $$V_{k+1}(s) = \max_{a} \left( \mathcal{R}_s^a + \gamma V_k(s') \right)$$
* **最佳路徑追蹤**：當價值矩陣收斂後，針對每個狀態提取最佳行動 $\pi^*(s)$。接著從起點出發，依循最佳行動箭頭前進直到抵達終點，並將此路徑標記為黃色高亮顯示。

---

## 📂 專案架構 (Project Structure)

本專案採用精簡且高內聚的雙檔案核心架構：

```text
GridWorld/
│
├── app.py                 # Flask 後端應用程式 (負責 API 路由與 RL 數學運算)
├── templates/
│   └── index.html         # 前端頁面 (負責 UI 渲染、點擊互動與 Fetch API 呼叫)
│
├── requirements.txt       # 環境套件依賴清單
└── README.md              # 專案說明文件
