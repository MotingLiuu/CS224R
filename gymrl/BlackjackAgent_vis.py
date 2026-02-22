from BlackjackAgent import BlackjackAgent
import gymnasium as gym
import time
import logging
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
logger = logging.getLogger(__name__)

n_episodes = 1000

env = gym.make("Blackjack-v1")

agent = BlackjackAgent.from_saved_file(env, "../data/blackjack_boltzmann_20260222_222500_q_table.pkl")

reward_history = []
for episode in range(n_episodes):
    obs, info = env.reset()
    episode_over = False
    total_reward = 0

    while not episode_over:
        action = agent.get_action_test(obs)
        logger.debug(f"Episode {episode + 1} | Observation: {obs} | Action taken: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        logger.debug(f"New observation: {obs} | Reward: {reward}")

        total_reward += reward
        episode_over = terminated or truncated

    logger.debug(f"Episode {episode + 1} finished! Total reward: {total_reward}")
    reward_history.append(total_reward)

logger.info(f"Average reward over {n_episodes} episodes: {sum(reward_history) / n_episodes}")

env.close()