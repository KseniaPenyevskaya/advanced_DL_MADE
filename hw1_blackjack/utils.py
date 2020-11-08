import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import defaultdict
from random import random, randint

def play(env, 
         strategy, 
         action_names=["hit", "stand"], 
         *args, 
         **kwargs):
    states = list()
    actions = list()
    rewards = list()
    env.reset()
    finish=False
    
    while finish==False:
        states.append(env._get_obs())
        action = strategy(env, *args, **kwargs)
        actions.append(action)
        new_state, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            finish = True
    env.close()
    return (states, actions, rewards)


def simple_strategy(env):
    action = 1 # hit
    if env._get_obs()[0] >= 19:
        action = 0 # stand
    return action


def eps_greedy_strategy(env, epsilon, Q):
    state = env._get_obs()
    if random() > epsilon:
        return Q[state].argmax()
    else:
        return randint(0, env.action_space.n - 1)


def average_reward(
    env,
    strategy,
    n_episodes=10,
    *args, 
    **kwargs):
    reward_sum = 0
    for iter in range(n_episodes):
        reward_sum += play(env, strategy, *args, **kwargs)[-1][-1]
    return reward_sum / n_episodes


def monte_carlo_control(
    env, 
    n_episodes=10, 
    gamma=0.3,
    decay=0.99,
    epsilon = 1):
    n = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n))
    returns = defaultdict(list)
    N = defaultdict(int)
    for iter in range(n_episodes):
        states, actions, rewards = play(env, eps_greedy_strategy, Q=Q, epsilon=epsilon)
        G = 0
        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            G = gamma * G + reward
            if (state, action) not in zip(states[:t], actions[:t]):
                N[(state, action)] += 1
                Q[state][action] += (G - Q[state][action]) / N[(state, action)]
        epsilon = max(epsilon * decay, 0.001)
    
    best_policy = lambda env, Q: eps_greedy_strategy(env, 0, Q)
    return best_policy, Q


def plot_monte_carlo(
    env,
    n_episodes_fit=100, 
    n_episodes_test=100000, 
    gammas = [0.1, 0.2, 0.3], 
    epsilon = 1,
    decay=0.999,
    verbose=False):
    
    all_rewards = []
    n = env.action_space.n
    for gamma in gammas:
        Q = defaultdict(lambda: np.zeros(n))
        N = defaultdict(int)
        
        rewards_gamma = []
        for iter in range(n_episodes_fit):
            if (iter + 1) % (n_episodes_fit // 20) == 0:
                best_policy = lambda env, Q: eps_greedy_strategy(env, 0, Q)
                rewards_gamma.append(average_reward(env, best_policy, n_episodes=n_episodes_test, Q=Q))
                if verbose:
                    print(f"Gamma: {gamma}    Episode: {iter+1}/{n_episodes_fit}    Reward: {rewards_gamma[-1]}.")
            states, actions, rewards = play(env, eps_greedy_strategy, Q=Q, epsilon=epsilon)
            G = 0
            for t in range(len(states) - 1, -1, -1):
                state = states[t]
                action = actions[t]
                reward = rewards[t]
                G = gamma * G + reward
                if (state, action) not in zip(states[:t], actions[:t]):
                    N[(state, action)] += 1
                    Q[state][action] += (G - Q[state][action]) / N[(state, action)]
            epsilon = max(epsilon * decay, 0.001)
        all_rewards.append(list(rewards_gamma))

    fig, ax = plt.subplots(1, figsize=(8, 6))
    fig.suptitle('Rewards', fontsize=15)

    for i in range(len(all_rewards)):
        ax.plot(all_rewards[i], label=f"Gamma: {gammas[i]}")
    plt.legend(loc="lower right",frameon=False)
    plt.show()



