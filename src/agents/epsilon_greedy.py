import numpy as np
from .base import BaseAgent


class EpsilonGreedyAgent(BaseAgent):
    def __init__(self, n_actions, alpha=0.1, epsilon=0.1):
        super().__init__(n_actions, alpha)
        self.epsilon = epsilon

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q)