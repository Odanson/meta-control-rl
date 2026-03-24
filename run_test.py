from src.envs.bandit import BanditEnv
from src.agents.meta_control import MetaControlAgent
from src.simulation.runner import run

env = BanditEnv(4, [0.2, 0.5, 0.7, 0.3], volatility=True)
agent = MetaControlAgent(4)

rewards, eps = run(env, agent, 500)

print("Total reward:", sum(rewards))
print("Final epsilon:", eps[-1])