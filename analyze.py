import matplotlib.pyplot as plt
import numpy as np

from src.envs.bandit import BanditEnv
from src.agents.epsilon_greedy import EpsilonGreedyAgent
from src.agents.meta_control import MetaControlAgent
from src.simulation.runner import run


def moving_average(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode="valid")


def run_experiment():
    n_steps = 1000

    env = BanditEnv(4, [0.2, 0.5, 0.7, 0.3], volatility=True)

    agents = {
        "fixed_eps": EpsilonGreedyAgent(4, epsilon=0.1),
        "meta_control": MetaControlAgent(4),
    }

    results = {}

    for name, agent in agents.items():
        env.reset()
        rewards, eps = run(env, agent, n_steps)

        results[name] = {
            "rewards": rewards,
            "eps": eps,
        }

    return results


def plot_results(results):
    plt.figure(figsize=(12, 5))

    # --- Rewards ---
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        r = moving_average(data["rewards"])
        plt.plot(r, label=name)

    plt.title("Smoothed Reward")
    plt.legend()

    # --- Epsilon ---
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        if data["eps"][0] is not None:
            plt.plot(data["eps"], label=name)

    plt.title("Exploration (epsilon)")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_experiment()
    plot_results(results)