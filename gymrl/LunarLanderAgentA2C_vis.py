from pathlib import Path
from datetime import datetime
import os
import numpy as np
import torch 
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import gymnasium as gym
from LunarLanderAgentA2C import A2C

checkpoint_path = Path("./LunarLanderA2C")
actor_weights_path = checkpoint_path / "actor_weights.h5"
critic_weights_path = checkpoint_path / "critic_weights.h5"

n_showcase_episodes = 3
env = gym.make("LunarLander-v3")
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

agent = A2C(obs_shape, action_shape, device="mps", critic_lr=0.001, actor_lr=0.005, n_envs=3)
agent.actor.load_state_dict(torch.load(actor_weights_path))
agent.critic.load_state_dict(torch.load(critic_weights_path))

for episode in range(n_showcase_episodes):
    print(f"starting episode {episode}...")

    env = gym.make(
        "LunarLander-v3",
        render_mode="human",
        gravity=np.clip(
            np.random.normal(loc=-10.0, scale=2.0), a_min=-11.99, a_max=-0.01
        ),
        enable_wind=np.random.choice([True, False]),
        wind_power=np.clip(
            np.random.normal(loc=15.0, scale=2.0), a_min=0.01, a_max=19.99
        ),
        turbulence_power=np.clip(
            np.random.normal(loc=1.5, scale=1.0), a_min=0.01, a_max=1.99
        ),
        max_episode_steps=500,
    )

    state, info = env.reset()

    # play one episode
    done = False
    while not done:
        # select an action A_{t} using S_{t} as input for the agent
        with torch.no_grad():
            action, _, _, _ = agent.select_action(state[None, :])

        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        state, reward, terminated, truncated, info = env.step(action.item())

        # update if the environment is done
        done = terminated or truncated

env.close()