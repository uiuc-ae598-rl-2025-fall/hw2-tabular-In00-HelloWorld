import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from collections import defaultdict

def epsilon_greedy(Q, state, n_actions, epsilon):
    if random.random() < epsilon: return random.randrange(n_actions)
    else:
        q_vals = Q[state]
        max_q = np.max(q_vals)
        actions = [a for a, q in enumerate(q_vals) if q == max_q]
        return random.choice(actions)

def e_greedy(Q, state, n_action, epsilon):
    policy = np.ones(n_action) * (epsilon / n_action)
    best_action = np.argmax(Q[state])
    policy[best_action] += 1 - epsilon
    return policy

def evaluate_policy(env, Q, gamma, num_eval_episodes=100):
    returns = []
    for _ in range(num_eval_episodes):
        state, info = env.reset()
        done = False
        G = 0
        step = 0
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            G += (gamma ** step) * reward
            step += 1
        returns.append(G)
    return np.mean(returns)


def MC(env, num_episodes=20000, gamma=0.95, epsilon=0.1, eval_interval=200, eval_episodes=100, seed=16): 
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)

    num_state = env.observation_space.n
    num_action = env.action_space.n
    Returns = defaultdict(list)
    Q = np.zeros((num_state, num_action))
    pi = np.ones((num_state, num_action)) * (1.0 / num_action)

    evaluation_returns = []
    Step_list = []
    t_steps = 0

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_data = []
        done = False

        while not done:
            action = np.random.choice(np.arange(num_action), p=pi[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state
            done = terminated or truncated
            t_steps += 1

        G = 0.0
        visited_sa = set()
        for t in reversed(range(len(episode_data))):
            s_t, a_t, r_t = episode_data[t]
            G = gamma * G + r_t
            if (s_t, a_t) not in visited_sa:
                visited_sa.add((s_t, a_t))
                Returns[(s_t, a_t)].append(G)
                Q[s_t, a_t] = np.mean(Returns[(s_t, a_t)])
                pi[s_t] = e_greedy(Q, s_t, num_action, epsilon)

        if episode % eval_interval == 0:
            avg_return = evaluate_policy(env, Q, gamma, eval_episodes)
            evaluation_returns.append(avg_return)
            Step_list.append(t_steps)
            print(f"[MC | Ep {episode}] Avg Return = {avg_return:.3f}")

    return Q, pi, {"timesteps": Step_list, "evaluation_returns": evaluation_returns}

def SARSA(env, num_episodes=20000, gamma=0.95, alpha=0.1, epsilon=0.1, eval_interval=200, eval_episodes=100, seed=16):
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    evaluation_returns = []
    Step_list = []
    t_steps = 0

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        state = state
        action = epsilon_greedy(Q, state, n_actions, epsilon)
        done = False

        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if not done:
                next_action = epsilon_greedy(Q, next_state, n_actions, epsilon)
            else:
                next_action = None

            target = reward
            if not done:
                target += gamma * Q[next_state, next_action]

            Q[state, action] += alpha * (target - Q[state, action])
            state = next_state
            action = next_action
            t_steps += 1

        if episode % eval_interval == 0:
            avg_return = evaluate_policy(env, Q, gamma, eval_episodes)
            evaluation_returns.append(avg_return)
            Step_list.append(t_steps)
            print(f"[SARSA | Ep {episode}] Avg Return = {avg_return:.3f}")

    return Q, {"timesteps": Step_list, "evaluation_returns": evaluation_returns}

def Q_learning(env, num_episodes=20000, gamma=0.95, alpha=0.1, epsilon=0.1, eval_interval=200, eval_episodes=100, seed=16):
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    evaluation_returns = []
    Step_list = []
    t_steps = 0

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        state = state
        done = False

        while not done:
            action = epsilon_greedy(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            target = reward
            if not done:
                target += gamma * np.max(Q[next_state])

            Q[state, action] += alpha * (target - Q[state, action])
            state = next_state
            t_steps += 1

        if episode % eval_interval == 0:
            avg_return = evaluate_policy(env, Q, gamma, eval_episodes)
            evaluation_returns.append(avg_return)
            Step_list.append(t_steps)
            print(f"[Q-learning | Ep {episode}] Avg Return = {avg_return:.3f}")

    return Q, {"timesteps": Step_list, "evaluation_returns": evaluation_returns}


def plot_results(results, title):
    plt.figure(figsize=(8,6))
    for algo, data in results.items():
        plt.plot(data["timesteps"], data["evaluation_returns"], label = algo, marker = 'o')
    plt.xlabel("Number of Time Steps")
    plt.ylabel("Evaluation Return")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_value_function(Q, env, title):
    V = np.max(Q, axis=1).reshape((4,4))
    grid_size = int(np.sqrt(env.observation_space.n))
    plt.figure(figsize=(5,5))
    plt.imshow(V.reshape((grid_size, grid_size)), cmap="viridis", origin="upper")
    plt.title(title)
    plt.colorbar(label="V(s)")
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center", color="black")
    plt.show()


def main_run(is_slippery):
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery)
    print(f"\n is_slippery={is_slippery}")

    Q_mc, _, mc_hist = MC(env)
    Q_sarsa, sarsa_hist = SARSA(env)
    Q_q, q_hist = Q_learning(env)

    results = {
        "MC": mc_hist,
        "SARSA": sarsa_hist,
        "Q-learning": q_hist
    }

    plot_results(results, f"FrozenLake: is_slippery={is_slippery}")
    plot_value_function(Q_mc, env, "Value Function of MC: is_slippery={is_slippery}")
    plot_value_function(Q_sarsa, env, f"Value Function of SARSA: is_slippery={is_slippery}")
    plot_value_function(Q_q, env, f"Value Function of Q-learning: is_slippery={is_slippery}")


main_run(is_slippery=False)
main_run(is_slippery=True)

