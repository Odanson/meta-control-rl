def run(env, agent, n_steps):
    rewards = []
    epsilons = []

    for _ in range(n_steps):
        action = agent.select_action()
        reward = env.step(action)

        agent.update(action, reward)

        rewards.append(reward)
        epsilons.append(getattr(agent, "epsilon", None))

    return rewards, epsilons