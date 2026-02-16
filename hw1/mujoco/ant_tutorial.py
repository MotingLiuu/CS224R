import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

import logging
logger = logging.getLogger(__name__)

env = gym.make("Ant-v4", render_mode = "rgb_array") 
reset_output = env.reset()
logger.debug("Reset output: %s", reset_output)

obs, _ = reset_output 

for _ in range(1000):
    frame = env.render()
    plt.imshow(frame)
    plt.axis('off')
    plt.clf()

    action = env.action_space.sample()
    step_output = env.step(action)
    obs, reward, terminated, truncated, info = step_output
    done = terminated or truncated
    if done:
        logger.debug("Episode ended with reward: %s", reward)
        break

    time.sleep(0.02)
    

env.close()