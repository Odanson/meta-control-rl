import numpy as np
from .base import BaseAgent


class MetaControlAgent(BaseAgent):
    def __init__(
        self,
        n_actions,
        alpha=0.1,
        epsilon_min=0.01,
        epsilon_max=0.5,
        uncertainty_scale=2.0,
        window=20,
    ):
        super().__init__(n_actions, alpha)

        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.uncertainty_scale = uncertainty_scale

        self.window = window
        self.pe_history = []

        self.epsilon = epsilon_min

    def _compute_uncertainty(self):
        if len(self.pe_history) < 2:
            return 0.0
        return np.mean(self.pe_history)

    def _update_epsilon(self):
        U = self._compute_uncertainty()
        eps = self.epsilon_min + self.uncertainty_scale * U
        self.epsilon = np.clip(eps, self.epsilon_min, self.epsilon_max)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q)

    def update(self, action, reward):
        prediction_error = abs(reward - self.Q[action])

        self.pe_history.append(prediction_error)
        if len(self.pe_history) > self.window:
            self.pe_history.pop(0)

        super().update(action, reward)
        self._update_epsilon()