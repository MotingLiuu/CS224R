from LunarLanderAgent import LunarLanderAgent
from LunarLander_vanillaPG import AgentVanillaPG
import gymnasium as gym
import time
import logging
from pathlib import Path
import numpy as np
import torch

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "vanilla_pg_VanillaPG_20260301_011851_env1_batch20_hidden128x2_actorlr0.001_update_steps500.pt"
n_episode = 10

env = gym.make(
    "LunarLander-v3",
    gravity=np.clip(
        np.random.normal(loc=-10.0, scale=1.0),
        a_min=-11.99,
        a_max=-0.01
    ),
    enable_wind=np.random.choice([True, False]),
    wind_power=np.clip(
        np.random.normal(loc=15.0, scale=1.0),
        a_min=0.01,
        a_max=19.9,
    ),
    turbulence_power=np.clip(
        np.random.normal(loc=1.5, scale=0.5),
        a_min=0.01,
        a_max=1.99,
    ),
    render_mode = "human",
)

agent = AgentVanillaPG.load(torch.device("cpu"), env, CHECKPOINT_PATH)
reward_history = []

for episode in range(n_episode):
    obs, info = env.reset()
    episode_over = False
    total_reward = 0

    while not episode_over:
        action = agent.get_action_test(obs)
        logger.debug(f"Episode {episode + 1} | Observation: {obs} | Action taken: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        logger.debug(f"New observation: {obs} | Reward: {reward}")

        total_reward += float(reward)
        episode_over = terminated or truncated

    logger.debug(f"Episode {episode + 1} finished! Total reward: {total_reward}")
    reward_history.append(total_reward)
    time.sleep(0.02)

logger.info(f"Average reward over {n_episode} episodes: {sum(reward_history) / n_episode}")

env.close()