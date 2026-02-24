from LunarLanderAgent import LunarLanderAgent
import gymnasium as gym
import time
import logging
from pathlib import Path

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "lunar_lander_agent_500.pt"
n_episode = 1000

env = gym.make("LunarLander-v3")
agent = LunarLanderAgent.load(env, CHECKPOINT_PATH)
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

logger.info(f"Average reward over {n_episode} episodes: {sum(reward_history) / n_episode}")

env.close()