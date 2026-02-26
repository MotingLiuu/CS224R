# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
import logging
import time

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
logger = logging.getLogger(__name__)

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("LunarLander-v3", render_mode="human")

logger.info(f"Action space: {env.action_space}")
logger.info(f"Sample action: {env.action_space.sample()}")

logger.info(f"Observation space: {env.observation_space}")
logger.info(f"Sample observation: {env.observation_space.sample()}")

# Reset environment to start a new episode
observation, info = env.reset()
# observation: what the agent can "see" - cart position, velocity, pole angle, etc.
# info: extra debugging information (usually not needed for basic learning)

print(f"Starting observation: {observation}")
# Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action: 0 = push cart left, 1 = push cart right
    action = env.action_space.sample()  # Random action for now - real agents will be smarter!
    time.sleep(0.05)  # Slow down the loop so we can see what's happening
    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    logger.info(f"Action taken: {action}")
    logger.info(f"New observation: {observation}")

    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()