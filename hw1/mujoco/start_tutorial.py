import gymnasium as gym
import numpy as np

import logging
logger = logging.getLogger(__name__)

env = gym.make("CartPole-v1", render_mode="human")
print(type(env))
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"render mode: {env.metadata['render_modes']}")

observation, info = env.reset()
print(f"Strarting observation: {observation}", f"info: {info}")

episode_over = False
total_reward = 0

while not episode_over:
    action = env.action_space.sample()
    observation, reward, episode_over, truncated, info = env.step(action)
    total_reward += reward
    episode_over = episode_over or truncated

print(f"Episode ended with total reward: {total_reward}")
env.close()