from Agent import Agent
from BanditEnv import BanditEnv

k = 10
epsilon = 0.1

agent = Agent(k, epsilon)
action = agent.select_action()
reward = 1.5  # 假設獲得的獎勵是 1.5
agent.update_q(action, reward)

print(f"Action selected: {action}, updated Q[{action}]: {agent.q_values[action]:.3f}")


