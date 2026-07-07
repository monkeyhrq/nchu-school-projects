from flask import Flask, render_template, request, jsonify
import numpy as np
import random

app = Flask(__name__)

# 定義四個移動方向
ACTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
GAMMA = 0.9
THETA = 1e-4

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/evaluate_random', methods=['POST'])
def evaluate_random():
    """HW1-2: 隨機策略與策略評估 (Policy Evaluation)"""
    data = request.json
    n = data['n']
    end = tuple(data['end']) if data.get('end') else None
    obstacles = [tuple(obs) for obs in data['obstacles']]
    
    V = np.zeros((n, n))
    policy = [['' for _ in range(n)] for _ in range(n)]
    
    # 1. 生成均勻隨機策略
    action_keys = list(ACTIONS.keys())
    for r in range(n):
        for c in range(n):
            if end and (r, c) == end:
                policy[r][c] = 'end'
            elif (r, c) in obstacles:
                policy[r][c] = 'obs'
            else:
                policy[r][c] = random.choice(action_keys)
                
    if not end:
        return jsonify({'error': '請先在網格上點擊設定終點 (紅色)！'})

    # 2. 策略評估 (Policy Evaluation)
    while True:
        delta = 0
        V_new = np.copy(V)
        for r in range(n):
            for c in range(n):
                if (r, c) == end or (r, c) in obstacles:
                    continue
                v_sum = 0
                for a_key in action_keys:
                    dr, dc = ACTIONS[a_key]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in obstacles:
                        reward = 10 if (nr, nc) == end else -1
                        v_sum += 0.25 * (reward + GAMMA * V[nr, nc])
                    else:
                        reward = -1  # 撞牆懲罰
                        v_sum += 0.25 * (reward + GAMMA * V[r, c])
                V_new[r, c] = v_sum
                delta = max(delta, abs(V_new[r, c] - V[r, c]))
        V = V_new
        if delta < THETA:
            break
            
    return jsonify({'v_matrix': V.tolist(), 'policy': policy, 'path': []})


@app.route('/api/value_iteration', methods=['POST'])
def value_iteration():
    """HW1-3: 價值迭代找出最佳政策 (Value Iteration)"""
    data = request.json
    n = data['n']
    start = tuple(data['start']) if data.get('start') else None
    end = tuple(data['end']) if data.get('end') else None
    obstacles = [tuple(obs) for obs in data['obstacles']]
    
    if not start or not end:
        return jsonify({'error': '請先在網格上點擊設定起點 (綠色) 與終點 (紅色)！'})
        
    V = np.zeros((n, n))
    policy = [['' for _ in range(n)] for _ in range(n)]
    
    # 1. 執行價值迭代 (Value Iteration)
    while True:
        delta = 0
        V_new = np.copy(V)
        for r in range(n):
            for c in range(n):
                if (r, c) == end or (r, c) in obstacles:
                    continue
                max_v = -float('inf')
                for a, (dr, dc) in ACTIONS.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in obstacles:
                        reward = 100 if (nr, nc) == end else -1
                        v = reward + GAMMA * V[nr, nc]
                    else:
                        reward = -1
                        v = reward + GAMMA * V[r, c]
                    if v > max_v:
                        max_v = v
                V_new[r, c] = max_v
                delta = max(delta, abs(V_new[r, c] - V[r, c]))
        V = V_new
        if delta < THETA:
            break

    # 2. 提取最佳策略 (Extract Optimal Policy)
    for r in range(n):
        for c in range(n):
            if (r, c) == end:
                policy[r][c] = 'end'
                continue
            if (r, c) in obstacles:
                policy[r][c] = 'obs'
                continue
            max_v = -float('inf')
            best_a = ''
            for a, (dr, dc) in ACTIONS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in obstacles:
                    reward = 100 if (nr, nc) == end else -1
                    v = reward + GAMMA * V[nr, nc]
                else:
                    reward = -1
                    v = reward + GAMMA * V[r, c]
                if v > max_v:
                    max_v = v
                    best_a = a
            policy[r][c] = best_a

    # 3. 追蹤最佳路徑 (Trace Optimal Path)
    path = []
    curr = start
    visited = set()
    while curr != end and curr not in visited:
        path.append(curr)
        visited.add(curr)
        r, c = curr
        a = policy[r][c]
        if not a or a in ('end', 'obs'): break
        dr, dc = ACTIONS[a]
        curr = (r + dr, c + dc)
        
    if curr == end:
        path.append(end)

    return jsonify({'v_matrix': V.tolist(), 'policy': policy, 'path': path})

if __name__ == '__main__':
    app.run(debug=True)