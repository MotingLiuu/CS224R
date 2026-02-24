from datetime import datetime
from pathlib import Path
from typing import Union
import gymnasium as gym
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from einops import rearrange, reduce, repeat

class QNet(nn.Module):
    def __init__(
        self,
        input_d: int,
        hidden_d: int,
        n_hidden_layer: int,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_d, hidden_d)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_d, hidden_d) for _ in range(n_hidden_layer)])
        self.output_layer = nn.Linear(hidden_d, 1)
    
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for hidden in self.hidden_layers:
            x = F.relu(hidden(x))
        return self.output_layer(x)

class LunarLanderAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        discount_factor: float = 0.95,
    ):
        self.env = env 
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.q_net = QNet(input_d=12, hidden_d=32, n_hidden_layer=2)
        self.target_net = QNet(input_d=12, hidden_d=32, n_hidden_layer=2)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.actions_onehot = np.eye(4)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.training_error = []

    def get_action_test(self, obs: tuple) -> int:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        action_tensor = torch.as_tensor(self.actions_onehot, dtype=torch.float32)
        input_tensor = torch.stack([torch.cat([obs_tensor, action]) for action in action_tensor])
        action_values_tensor = self.q_net(input_tensor)
        action_values_tensor = rearrange(action_values_tensor, "b d -> (b d)")
        action = int(torch.argmax(action_values_tensor))
        return action

    def get_action_boltzmann(self, obs: tuple) -> int:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        action_tensor = torch.as_tensor(self.actions_onehot, dtype=torch.float32)
        input_tensor = torch.stack([torch.cat([obs_tensor, action]) for action in action_tensor])
        action_values_tensor = self.q_net(input_tensor)
        action_values_tensor = rearrange(action_values_tensor, "b d -> (b d)")
        exp_tensor = torch.exp(action_values_tensor - torch.max(action_values_tensor))
        action_probs = exp_tensor / torch.sum(exp_tensor)
        action_dist = Categorical(action_probs)
        action_tensor = action_dist.sample()
        action = int(action_tensor)
        return action

    def update_q_value(
        self,
        obs: tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple,
    ):
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32)
        next_action_tensor = torch.as_tensor(self.actions_onehot, dtype=torch.float32)
        next_input_tensor = torch.stack([torch.cat([next_obs_tensor, action]) for action in next_action_tensor])
        future_q_value = (not terminated) * self.discount_factor * (torch.max(self.target_net(next_input_tensor).detach()))
        target = reward + future_q_value

        obs_tensor, action_tensor = torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(self.actions_onehot[action],dtype=torch.float32)
        input_tensor = torch.cat([obs_tensor, action_tensor])
        loss = F.mse_loss(self.q_net(input_tensor), torch.as_tensor(target))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_error.append(loss.detach().numpy())
    
    def update_target_hard(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def update_target_soft(self, tau: float):
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, checkpoint_path: Union[str, Path]) -> Path:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "agent_hparams": {
                "learning_rate": self.lr,
                "discount_factor": self.discount_factor,
            },
            "qnet_hparams": {
                "input_d": self.q_net.input_layer.in_features,
                "hidden_d": self.q_net.input_layer.out_features,
                "n_hidden_layer": len(self.q_net.hidden_layers),
            },
            "actions_onehot": self.actions_onehot,
            "q_net_state_dict": self.q_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_error": self.training_error,
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    @classmethod
    def load(cls, env: gym.Env, checkpoint_path: Union[str, Path], map_location: str = "cpu"):
        checkpoint_path = Path(checkpoint_path)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

        agent_hparams = checkpoint["agent_hparams"]
        agent = cls(
            env=env,
            learning_rate=agent_hparams["learning_rate"],
            discount_factor=agent_hparams["discount_factor"],
        )

        qnet_hparams = checkpoint["qnet_hparams"]
        agent.q_net = QNet(
            input_d=qnet_hparams["input_d"],
            hidden_d=qnet_hparams["hidden_d"],
            n_hidden_layer=qnet_hparams["n_hidden_layer"],
        )
        agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        agent.q_net.eval()

        agent.actions_onehot = checkpoint["actions_onehot"]
        agent.optimizer = optim.Adam(agent.q_net.parameters(), lr=agent.lr)
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.training_error = checkpoint["training_error"]
        return agent


if __name__ == "__main__":
    import wandb
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    config = {
        "learning_rate": 1e-3,
        "n_episode": 2000,
        "discount_factor": 0.99,
        "sample_method": "boltzmann",
        "target_update_method": "soft",
        "target_update_step": 500,
        "target_update_tau": 5e-3,
    }

    if config["sample_method"] == "boltzmann":
        run_name = config["sample_method"] + datetime.now().strftime("_%Y%m%d_%H%M%S")
    else:
        run_name = datetime.now().strftime("LunarLander_%Y%m%d_%H%M%S")
    wandb.init(
        project="gymrl",
        name=run_name,
        config=config,
    )
    
    n_episode = config["n_episode"]
    learning_rate = config["learning_rate"]
    if config["target_update_method"] == "hard":
        target_update_step = config["target_update_step"]
    else:
        target_update_tau = config["target_update_tau"]
    total_step = 0
    checkpoint_dir = Path("checkpoints")
    checkpoint_path = checkpoint_dir / f"{config["sample_method"]}_{config["n_episode"]}.pt"

    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)

    agent = LunarLanderAgent(env=env, learning_rate=learning_rate)
    
    for episode in range(n_episode):
        obs, info = env.reset()
        episode_over = False
        episode_reward = 0.0
        td_error_start_idx = len(agent.training_error)

        while not episode_over:
            action = agent.get_action_boltzmann(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update_q_value(obs, action, float(reward), terminated, next_obs)
            if config["target_update_method"] == "hard":
                if total_step % target_update_step == 0:
                    agent.update_target_hard()
            else:
                agent.update_target_soft(target_update_tau)
            total_step += 1
                
            obs = next_obs
            episode_over = terminated or truncated
            episode_reward += float(reward)

        logger.info(f"Episode {episode} finished! Total reward: {episode_reward}")
        episode_td_errors = agent.training_error[td_error_start_idx:]
        mean_td_error = float(np.mean(episode_td_errors)) if episode_over else 0.0
    
        wandb.log({
            "train/episode_reward": episode_reward,
            "train/mean_td_error": mean_td_error,
            "episode": episode,
        })

    saved_path = agent.save(checkpoint_path)
    logger.info(f"Checkpoint saved to: {saved_path}")






        