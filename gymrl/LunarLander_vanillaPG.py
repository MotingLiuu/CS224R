from pathlib import Path
from typing import Union
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from einops import rearrange, reduce, repeat

class Agent():
    def __init__(
        self,
        device: torch.device,
        discount_factor: float,
        actor_lr: float,
        in_features: int,
        out_features: int,
        hidden_features: int,
        n_hidden_layers: int,
    ):
        self.device = device
        self.discount_factor = discount_factor
        self.actor_lr = actor_lr
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.n_hidden_layers = n_hidden_layers

        self.actor = nn.Sequential(
            nn.Sequential(nn.Linear(self.in_features, self.hidden_features), nn.ReLU()),
            *[nn.Sequential(nn.Linear(self.hidden_features, self.hidden_features), nn.ReLU()) for _ in range(self.n_hidden_layers)],
            nn.Linear(self.hidden_features, self.out_features)
        ).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

    def get_vanilla_loss(self, actions_log_probs: torch.Tensor, rewards: np.ndarray, masks: np.ndarray) -> torch.Tensor:
        n_steps = rewards.shape[-1]
        advantages = np.zeros_like(rewards)
        for index in range(n_steps-1, -1, -1):
            if index == n_steps - 1:
                advantages[...,index] = rewards[...,index]
            else:
                advantages[...,index] = rewards[...,index] + (1 - masks[...,index]) * advantages[...,index+1] * self.discount_factor
        loss = -torch.mean(actions_log_probs * torch.as_tensor(advantages, dtype=torch.float32, device=self.device))
        
        return loss
    
    def update_actor(self, loss: torch.Tensor):
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions_logits_tensor = self.actor(obs_tensor)
        actions_dist = Categorical(logits=actions_logits_tensor)
        actions = actions_dist.sample()
        actions = actions.cpu().numpy()

        return actions

    def get_log_prob(self, obs: np.ndarray, actions: np.ndarray) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions_logits_tensor = self.actor(obs_tensor)
        actions_dist = Categorical(logits=actions_logits_tensor)
        actions_log_probs_tensor = actions_dist.log_prob(actions_tensor)

        return actions_log_probs_tensor

    def save_agent(self, checkpoint_path: Union[str, Path]) -> Path:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "agent_hparams": {
                "device": self.device,
                "discount_factor": self.discount_factor,
                "actor_lr": self.actor_lr,
                "in_features": self.in_features,
                "out_features": self.out_features,
                "hidden_features": self.hidden_features,
                "n_hidden_layers": self.n_hidden_layers,
            },
            "actor_state_dict": self.actor.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    @classmethod
    def load(cls, device: torch.device, env: gym.Env, checkpoint_path: Union[str, Path]):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent_hparams = checkpoint["agent_hparams"]
        agent = cls(
            device=device,
            discount_factor=agent_hparams["discount_factor"],
            actor_lr=agent_hparams["actor_lr"],
            in_features=agent_hparams["in_features"],
            out_features=agent_hparams["out_features"],
            hidden_features=agent_hparams["hidden_features"],
            n_hidden_layers=agent_hparams["n_hidden_layers"],
        )
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.actor_optim.load_state_dict(checkpoint["actor_optim_state_dict"])

        return agent

    
if __name__ == "__main__":
    import logging
    import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)
    from datetime import datetime
    import wandb
    
    config = {
        "n_envs": 2,
        "n_update_steps": 1000,
        "batch_size": 10,
        "in_features": 8,
        "out_features": 4,
        "hidden_features": 16,
        "n_hidden_layers": 2,
        "actor_lr": 1e-3,
        "discount_factor": 0.99,
        "device": "mps",
    }

    run_name = "VanillaPG" + datetime.now().strftime("_%Y%m%d_%H%M%S") + f"_env{config['n_envs']}_batch{config['batch_size']}_hidden{config['hidden_features']}x{config['n_hidden_layers']}_actorlr{config['actor_lr']}_update_steps{config['n_update_steps']}"
    wandb.init(
        project="gymrl",
        name=run_name,
        config=config,
    )
    
    envs = gym.vector.SyncVectorEnv(
        [
            lambda: gym.make(
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
                )
            )
            for i in range(config["n_envs"])
        ]
    )
    device = torch.device(config["device"])
    n_envs = config["n_envs"]
    batch_size = config["batch_size"]

    envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    agent = Agent(
        device=device,
        discount_factor=config["discount_factor"], 
        actor_lr=config["actor_lr"],
        in_features=config["in_featues"],
        out_features=config["out_features"],
        hidden_features=config["hidden_features"],
        n_hidden_layers=config["n_hidden_layers"],
    )
    
    obs, info = envs.reset()
    
    with logging_redirect_tqdm():
        for step in range(config["n_update_steps"]):
            batch_obs_list, batch_actions_list, batch_rewards_list = [], [], []
            batch_done_list = []
            cur_episodes = 0
            obs_list, actions_list, rewards_list = [[] for _ in range(n_envs)], [[] for _ in range(n_envs)], [[] for _ in range(n_envs)]
            while cur_episodes < batch_size: 
                actions = agent.get_actions(obs=obs)
                next_obs, rewards, termindated, truncated, infos = envs_wrapper.step(actions=actions)
                for i in range(n_envs):
                    obs_list[i].append(obs[i])
                    actions_list[i].append(actions[i])
                    rewards_list[i].append(rewards[i])
                if "episode" in infos:
                    for i in range(len(infos["_episode"])):
                        if infos["_episode"][i]:
                            episode_reward = infos["episode"]["r"][i]
                            episode_length = infos["episode"]["l"][i]
                            
                            logging.info(f"episode reward: {episode_reward}")
                            logging.info(f"episode length: {episode_length}")

                            wandb.log(
                                {
                                    "train/reward": episode_reward,
                                    "train/length": episode_length,
                                }
                            )
                            
                            batch_obs_list.extend(obs_list[i])
                            batch_actions_list.extend(actions_list[i])
                            batch_rewards_list.extend(rewards_list[i])
                            batch_done_list.extend([0 for _ in range(len(obs_list[i]) - 1)] + [1])
                            cur_episodes += 1

                            obs_list[i], actions_list[i], rewards_list[i] = [], [], []
                obs = next_obs

            batch_obs_array, batch_actions_array, batch_rewards_array, batch_done_array = np.array(batch_obs_list), np.array(batch_actions_list), np.array(batch_rewards_list), np.array(batch_done_list)
            batch_actions_log_probs = agent.get_log_prob(obs=batch_obs_array, actions=batch_actions_array)
            loss = agent.get_vanilla_loss(actions_log_probs=batch_actions_log_probs, rewards=batch_rewards_array, masks=batch_done_array)
            agent.update_actor(loss)

            logging.info(f"Step {step + 1} | Loss: {loss.item()}")
            wandb.log(
                {
                    "train/loss": loss.item(),
                }
            )
    agent.save_agent(checkpoint_path=Path(__file__).parent / "checkpoints" / f"vanilla_pg_{run_name}.pt")
