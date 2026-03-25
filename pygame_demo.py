import pygame
import numpy as np

from src.envs.bandit import BanditEnv
from src.agents.meta_control import MetaControlAgent
from src.agents.epsilon_greedy import EpsilonGreedyAgent

# --- Config ---
WIDTH, HEIGHT = 800, 600
FPS = 30
N_ARMS = 4
RADIUS = 200
CENTER = np.array([WIDTH // 2, HEIGHT // 2])

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# --- Setup environment and agent ---
env = BanditEnv(N_ARMS, [0.2, 0.5, 0.7, 0.3], volatility=True, switch_interval=100)
agent = MetaControlAgent(N_ARMS)

# agent = EpsilonGreedyAgent(N_ARMS, epsilon=0.1)  # toggle baseline

# --- Precompute arm positions ---
angles = np.linspace(0, 2 * np.pi, N_ARMS, endpoint=False)
arm_positions = [
    CENTER + RADIUS * np.array([np.cos(a), np.sin(a)])
    for a in angles
]

agent_pos = CENTER.astype(float).copy()
target_pos = CENTER.astype(float).copy()
current_action = None
reward_flash = 0
cumulative_reward = 0
step_count = 0

running = True
while running:
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                # Force environment change
                env._shuffle_rewards()

    # --- Select new action if reached target ---
    if np.linalg.norm(agent_pos - target_pos) < 5:
        action = agent.select_action()
        current_action = action
        target_pos = arm_positions[action]

        reward = env.step(action)
        agent.update(action, reward)

        cumulative_reward += reward
        reward_flash = 10 if reward > 0 else 0
        step_count += 1

    # --- Move agent ---
    direction = target_pos - agent_pos
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)
    agent_pos += direction * 5

    # --- Draw arms ---
    for i, pos in enumerate(arm_positions):
        color = (100, 100, 100)
        if i == current_action:
            color = (0, 200, 255)
        pygame.draw.circle(screen, color, pos.astype(int), 30)

    # --- Draw agent ---
    agent_color = (255, 255, 255)
    if reward_flash > 0:
        agent_color = (0, 255, 0)
        reward_flash -= 1

    pygame.draw.circle(screen, agent_color, agent_pos.astype(int), 10)

    # --- Text ---
    epsilon = getattr(agent, "epsilon", 0.0)

    texts = [
        f"Steps: {step_count}",
        f"Reward: {cumulative_reward}",
        f"Epsilon: {epsilon:.3f}",
        "Press C to trigger change",
    ]

    for i, t in enumerate(texts):
        img = font.render(t, True, (255, 255, 255))
        screen.blit(img, (10, 10 + i * 25))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
