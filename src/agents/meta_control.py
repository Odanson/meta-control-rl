import numpy as np
from .base import BaseAgent


class MetaControlAgent(BaseAgent):
    def __init__(
        self,
        n_actions,
        alpha=0.1,
        epsilon_min=0.05,
        epsilon_max=0.5,
        smooth_alpha=0.2,
        uncertainty_scale=0.35,
        window=20,
    ):
        super().__init__(n_actions, alpha)

        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.uncertainty_scale = uncertainty_scale

        self.window = window
        self.pe_history = []

        self.epsilon = epsilon_min
        self.last_action = 0

        self.epsilon_smooth = epsilon_min
        self.smooth_alpha = smooth_alpha

    def _compute_uncertainty(self, action):
        # Action-specific uncertainty: high when this action is under-sampled
        return 1 / (self.N[action] + 1)

    def _value_uncertainty(self):
        if self.n_actions < 2:
            return 0.0
        sorted_Q = np.sort(self.Q)
        gap = sorted_Q[-1] - sorted_Q[-2]

        U = 1 / (gap + 1e-5)
        return np.clip(U, 0, 2.0)

    def _update_epsilon(self):
        # Combine count-based and value-based uncertainty
        U_count = self._compute_uncertainty(self.last_action)
        U_value = self._value_uncertainty()
        U = 0.95 * U_count + 0.05 * U_value

        raw_eps = self.epsilon_min + self.uncertainty_scale * U
        raw_eps = np.clip(raw_eps, self.epsilon_min, self.epsilon_max)

        # Smooth epsilon over time
        self.epsilon_smooth = (
            (1 - self.smooth_alpha) * self.epsilon_smooth
            + self.smooth_alpha * raw_eps
        )

        self.epsilon = self.epsilon_smooth

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q)

    def update(self, action, reward):
        self.last_action = action

        prediction_error = abs(reward - self.Q[action])

        # Surprise-triggered forgetting
        if prediction_error > 0.7:
            self.N *= 0.97
        else:
            self.N *= 0.999

        super().update(action, reward)
        self._update_epsilon()