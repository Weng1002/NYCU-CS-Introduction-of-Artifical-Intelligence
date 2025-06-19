# main.py
import numpy as np
import matplotlib.pyplot as plt
from BanditEnv import BanditEnv
from Agent import Agent
from tqdm import tqdm

def run_experiment(epsilon, runs=2000, steps=1000, k=10):
    rewards = np.zeros((runs, steps))
    optimal_actions = np.zeros((runs, steps))

    for run in tqdm(range(runs), desc=f"ε = {epsilon}"):
        env = BanditEnv(k)
        agent = Agent(k, epsilon)
        env.reset()
        agent.reset()

        optimal_action = int(np.argmax(env.true_means))

        for t in range(steps):
            action = agent.select_action()
            reward = env.step(action)
            agent.update_q(action, reward)

            rewards[run, t] = reward
            if action == optimal_action:
                optimal_actions[run, t] = 1

    avg_rewards = rewards.mean(axis=0)
    optimal_action_rate = optimal_actions.mean(axis=0)
    return avg_rewards, optimal_action_rate

def plot_results(results, steps=1000, filename="part3_stationary.png"):
    plt.figure(figsize=(12, 5))

    # Average Reward
    plt.subplot(1, 2, 1)
    for label, (reward, _) in results.items():
        plt.plot(reward, label=f"ε={label}")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward over Time")
    plt.legend()

    # Optimal Action Rate
    plt.subplot(1, 2, 2)
    for label, (_, opt) in results.items():
        plt.plot(opt * 100, label=f"ε={label}")  
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.title("Optimal Action Percentage")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)  
    plt.show()


if __name__ == "__main__":
    epsilons = [0, 0.1, 0.01]
    results = {}

    for eps in epsilons:
        avg_reward, opt_action_rate = run_experiment(eps)
        results[eps] = (avg_reward, opt_action_rate)

    plot_results(results, filename="part3_stationary.png")
