import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from cliff_walking import CliffWalkingEnv
from agent import QLearningAgent, SarsaAgent

def train(agent, env, episodes):
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False
        
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)
            
            agent.update(state, action, reward, next_state, next_action, done)
            
            # The clip to -100 per episode is often used in these plots or the axis is limited
            total_reward += reward
            state = next_state
            action = next_action
            
        rewards[ep] = max(-100, total_reward)
    return rewards

def draw_policy(agent, ax, title):
    env = CliffWalkingEnv()
    rows, cols = env.rows, env.cols
    
    # Draw Grid
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.grid(color='k', linestyle='-', linewidth=2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=16, pad=15)
    
    # Fill Cliff
    for c in range(1, 11):
        rect = Rectangle((c-0.5, 3-0.5), 1, 1, color='lightblue')
        ax.add_patch(rect)
        
    ax.text(5.5, 3, "Cliff", ha="center", va="center", fontsize=14)
    ax.text(0, 3, "Start\n", ha="center", va="bottom", fontsize=12)
    ax.text(11, 3, "Goal", ha="center", va="center", fontsize=12)
    
    # Arrow drawing
    # Actions: 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
    arrow_len = 0.35
    dx = {0: 0, 1: arrow_len, 2: 0, 3: -arrow_len}
    dy = {0: -arrow_len, 1: 0, 2: arrow_len, 3: 0} # y-axis is inverted
    
    for r in range(rows):
        for c in range(cols):
            if (r == 3 and 1 <= c <= 10) or (r == 3 and c == 11):
                continue
                
            q_values = agent.q_table[r, c]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            
            # Prefer actions that match the teacher's plot (e.g., Q-learning: right arrows just above cliff)
            # If there's a tie, we pick the best visually. Or just use the first.
            # To strictly match teacher's plot we should let it fully learn or just pick any.
            a = best_actions[0] 
            
            x, y = c, r
            sx = x - dx[a]*0.5
            sy = y - dy[a]*0.5
            ax.arrow(sx, sy, dx[a], dy[a], head_width=0.15, head_length=0.15, fc='k', ec='k', width=0.015)

    # Trace greedy path and draw dashed line
    state = env.start_state
    
    # To draw the path exactly like the image, the line needs to be slightly offset
    path_x = []
    path_y = []
    
    done = False
    steps = 0
    # Add start offset
    path_x.append(state[1] - 0.2)
    path_y.append(state[0] + 0.2)
    path_x.append(state[1])
    path_y.append(state[0])

    while not done and steps < 100:
        r, c = state
        a = np.argmax(agent.q_table[r, c])
        next_state, _, done = env.step(a)
        path_x.append(next_state[1])
        path_y.append(next_state[0])
        state = next_state
        steps += 1
        
    ax.plot(path_x, path_y, color='tab:blue', linestyle='--', linewidth=3)
    
    # Add border box
    rect = Rectangle((-0.5, -0.5), cols, rows, fill=False, color='k', linewidth=4)
    ax.add_patch(rect)

def main():
    env = CliffWalkingEnv()
    episodes = 500
    runs = 50
    alpha = 0.5  # Match teacher's plot
    gamma = 1.0  # Match Sutton & Barto completely
    epsilon = 0.1
    
    q_rewards_total = np.zeros((runs, episodes))
    sarsa_rewards_total = np.zeros((runs, episodes))
    
    print(f"Running experiments averaged over {runs} runs with {episodes} episodes...")
    for r in range(runs):
        print(f"Run {r+1}/{runs}", end='\r')
        q_agent = QLearningAgent(env.rows, env.cols, env.action_space, alpha=alpha, gamma=gamma, epsilon=epsilon)
        q_rewards_total[r] = train(q_agent, env, episodes)
        
        sarsa_agent = SarsaAgent(env.rows, env.cols, env.action_space, alpha=alpha, gamma=gamma, epsilon=epsilon)
        sarsa_rewards_total[r] = train(sarsa_agent, env, episodes)
    print("\nExperiments complete!")
        
    sarsa_avg = np.mean(sarsa_rewards_total, axis=0)
    q_avg = np.mean(q_rewards_total, axis=0)
    
    # 1. Plotting Reward Curve like Teacher's format
    plt.figure(figsize=(10, 7))
    plt.plot(sarsa_avg, color='tab:cyan', linewidth=2, label='Sarsa')
    plt.plot(q_avg, color='tab:red', linewidth=2, label='Q-learning')
    
    plt.ylim(-100, 0)
    plt.yticks(np.arange(-100, 1, 20))
    plt.grid(True, linestyle='-', linewidth=0.5)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Reward Sum for Episode', fontsize=12)
    plt.title('Sarsa Vs. Q-Learning Cliff Walking\nEpsilon=0.1, Alpha=0.5\n(averaged over 50 runs)', fontsize=14)
    
    # Legend settings
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig('learning_curve_50_runs.png', dpi=300)
    print("Saved learning_curve_50_runs.png")
    
    # We train one full final agent to get a stable policy to draw
    final_q = QLearningAgent(env.rows, env.cols, env.action_space, alpha=alpha, gamma=gamma, epsilon=epsilon)
    train(final_q, env, eps=5000) # train longer for a super stable policy visualization
    
    final_sarsa = SarsaAgent(env.rows, env.cols, env.action_space, alpha=alpha, gamma=gamma, epsilon=epsilon)
    train(final_sarsa, env, eps=5000)

    # 2. Plotting Policies like Teacher's format
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    draw_policy(final_q, axes[0], "Q-learning policy")
    draw_policy(final_sarsa, axes[1], "Sarsa policy")
    
    plt.tight_layout()
    plt.savefig('policies_visualization.png', dpi=300)
    print("Saved policies_visualization.png")

# Overload train arguments locally for the single final plot
def train(agent, env, eps=500):
    rewards = np.zeros(eps)
    for ep in range(eps):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False
        
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)
            
            agent.update(state, action, reward, next_state, next_action, done)
            
            total_reward += reward
            state = next_state
            action = next_action
            
        rewards[ep] = max(-100, total_reward)
    return rewards

if __name__ == "__main__":
    main()
