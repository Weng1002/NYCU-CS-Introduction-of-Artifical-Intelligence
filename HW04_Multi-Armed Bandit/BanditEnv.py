import numpy as np

class BanditEnv:
    def __init__(self, k, stationary=True):
        self.k = k
        self.stationary = stationary  # 是否為 stationary 環境，True 為 stationary，False 為 non-stationary (step 4)
        self.time = 0
        self.reset()

    def reset(self):
        self.true_means = np.random.normal(0, 1, self.k)
        self.action_history = []
        self.reward_history = []
        self.time = 0

    def step(self, action):
        reward = np.random.normal(self.true_means[action], 1.0) # 標準差為 1
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.time += 1

        # 若是 non-stationary 環境，每一步更新 true means（小幅隨機漫步）
        if not self.stationary:
            self.true_means += np.random.normal(0, 0.01, self.k)

        return reward

    def export_history(self):
        return self.action_history, self.reward_history