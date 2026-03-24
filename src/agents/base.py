import numpy as np


class BaseAgent:
    def __init__(self, n_actions, alpha=0.1):
        self.n_actions = n_actions
        self.alpha = alpha

        self.Q = np.zeros(n_actions)
        self.N = np.zeros(n_actions)

    def select_action(self):
        raise NotImplementedError

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += self.alpha * (reward - self.Q[action])