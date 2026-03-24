from src.envs.bandit import BanditEnv

def test_bandit_runs():
    env = BanditEnv(4, [0.2, 0.5, 0.7, 0.3])
    r = env.step(0)
    assert r in [0.0, 1.0]