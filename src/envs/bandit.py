import numpy as np


class BanditEnv:
    def __init__(self, n_arms, reward_probs, volatility=False, switch_interval=100, seed=None):
        self.n_arms = n_arms
        self.reward_probs = np.array(reward_probs)
        self.volatility = volatility
        self.switch_interval = switch_interval
        self.rng = np.random.default_rng(seed)

        self.t = 0

    def step(self, action: int):
        reward = self.rng.random() < self.reward_probs[action]
        self.t += 1

        if self.volatility and self.t % self.switch_interval == 0:
            self._shuffle_rewards()

        return float(reward)

    def _shuffle_rewards(self):
        self.reward_probs = self.rng.permutation(self.reward_probs)

    def reset(self):
        self.t = 0