import numpy as np
from tqdm import tqdm

from src.envs.bandit import BanditEnv
from src.agents.epsilon_greedy import EpsilonGreedyAgent
from src.agents.meta_control import MetaControlAgent
from src.simulation.runner import run


def evaluate(n_runs=30, n_steps=1000):
    results = {
        "fixed": [],
        "meta": []
    }

    for seed in tqdm(range(n_runs)):
        # Fixed epsilon agent
        env = BanditEnv(4, [0.2, 0.5, 0.7, 0.3], volatility=True, seed=seed)
        agent = EpsilonGreedyAgent(4, epsilon=0.1)
        rewards, _ = run(env, agent, n_steps)
        results["fixed"].append(sum(rewards))

        # Meta-control agent
        env = BanditEnv(4, [0.2, 0.5, 0.7, 0.3], volatility=True, seed=seed)
        agent = MetaControlAgent(4)
        rewards, eps = run(env, agent, n_steps)
        results["meta"].append(sum(rewards))

    return results


def summarize(results):
    for key, vals in results.items():
        vals = np.array(vals)
        print(f"\n=== {key.upper()} ===")
        print(f"Mean reward: {vals.mean():.2f}")
        print(f"Std reward:  {vals.std():.2f}")
        print(f"Min / Max:   {vals.min():.2f} / {vals.max():.2f}")


if __name__ == "__main__":
    results = evaluate(n_runs=30, n_steps=1000)
    summarize(results)
