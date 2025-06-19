import numpy as np
import matplotlib.pyplot as plt
from BanditEnv import BanditEnv
from Agent import Agent
from tqdm import tqdm

def run_nonstationary_experiment(epsilon, step_size=None, runs=2000, steps=10000, k=10):
    rewards = np.zeros((runs, steps))
    optimal_actions = np.zeros((runs, steps))

    label = "sample-average" if step_size is None else f"constant α={step_size}"

    for run in tqdm(range(runs), desc=f"{label}"):
        env = BanditEnv(k, stationary=False)
        agent = Agent(k, epsilon, step_size=step_size)
        env.reset()
        agent.reset()

        for t in range(steps):
            action = agent.select_action()
            reward = env.step(action)
            agent.update_q(action, reward)

            rewards[run, t] = reward
            optimal_action = int(np.argmax(env.true_means))
            if action == optimal_action:
                optimal_actions[run, t] = 1

    avg_rewards = rewards.mean(axis=0)
    opt_action_rate = optimal_actions.mean(axis=0)
    return avg_rewards, opt_action_rate

def plot_results(results, filename="part7_step_size_comparison.png"):
    plt.figure(figsize=(12, 5))

    # Average Reward
    plt.subplot(1, 2, 1)
    for label, (reward, _) in results.items():
        plt.plot(reward, label=label)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Avg Reward (Non-Stationary)")
    plt.legend()

    # Optimal Action Percentage
    plt.subplot(1, 2, 2)
    for label, (_, opt_rate) in results.items():
        plt.plot(opt_rate * 100, label=label)
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.title("Optimal Action Rate (Non-Stationary)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    results = {}

    results["sample-average"] = run_nonstationary_experiment(epsilon=0.1, step_size=None)
    results["constant α=0.1"] = run_nonstationary_experiment(epsilon=0.1, step_size=0.1)

    plot_results(results, filename="part7_step_size_comparison.png")
