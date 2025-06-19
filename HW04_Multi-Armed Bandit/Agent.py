import numpy as np
import random

class Agent:
    def __init__(self, k, epsilon, step_size=None):
        self.k = k
        self.epsilon = epsilon
        self.step_size = step_size # Step 6
        self.reset()

    def reset(self):
        self.q_values = np.zeros(self.k)
        self.action_counts = np.zeros(self.k)
        self.time = 0

    # ε-greedy policy
    def select_action(self):  
        if random.random() < self.epsilon:
            return random.randint(0, self.k - 1)  # Exploration：隨機選一個 action，不斷嘗試
        else:
            return int(np.argmax(self.q_values))  # Exploitation：選最大 Q 的 action，選最佳

    def update_q(self, action, reward):
        self.action_counts[action] += 1
        if self.step_size is not None:   # Step 6
            alpha = self.step_size
        else:
            alpha = 1.0 / self.action_counts[action] # Sample-Average 更新法

        self.q_values[action] += alpha * (reward - self.q_values[action]) # Q(a)←Q(a)+α⋅(R−Q(a))
        self.time += 1