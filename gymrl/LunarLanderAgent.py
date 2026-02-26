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
from gymnasium import spaces
from einops import rearrange, reduce, repeat

class ReplayBuffer():
    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
        n_envs: int,
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.pos = 0
        self.full = False
        
        self.observations = np.zeros((self.buffer_size, self.n_envs, self.observation_dim))
        self.actions = np.zeros((self.buffer_size, self.n_envs))
        self.rewards = np.zeros((self.buffer_size, self.n_envs))
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, self.observation_dim))
        self.done = np.zeros((self.buffer_size, self.n_envs))
        
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ):
        self.observations[self.pos] = obs
        self.next_observations[self.pos]= next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.done[self.pos] = done

        if self.pos >= self.buffer_size - 1:
            self.full = True
        self.pos += 1
        self.pos = self.pos % self.buffer_size
    
    def sample(self, batch_size: int):
        upper_bound = self.buffer_size if self.full else self.pos + 1
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        env_indices = np.random.randint(0, high=self.n_envs, size=batch_size)
        data = {
            "obs": self.observations[batch_inds, env_indices],
            "actions": self.actions[batch_inds, env_indices],
            "rewards": self.rewards[batch_inds, env_indices],
            "next_obs": self.next_observations[batch_inds, env_indices],
            "done": self.done[batch_inds, env_indices],
        }
        return data


class QNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_d: int,
        n_actions: int,
        n_hidden_layers: int,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_d = hidden_d
        self.n_actions = n_actions
        self.n_hidden_layers = n_hidden_layers
        
        self.input_layer = nn.Linear(self.n_features, self.hidden_d)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_d, self.hidden_d) for _ in range(self.n_hidden_layers)])
        self.output_layer = nn.Linear(self.hidden_d, self.n_actions)
    
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for hidden in self.hidden_layers:
            x = F.relu(hidden(x))
        return self.output_layer(x)

class LunarLanderAgent:
    def __init__(
        self,
        device: torch.device,
        learning_rate: float,
        epsilon: float,
        min_epsilon: float,
        decay_steps: float,
        discount_factor: float = 0.99,
    ):
        self.device = device
        self.lr = learning_rate
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_steps = decay_steps
        self.discount_factor = discount_factor
        self.q_net = QNet(n_features=8, hidden_d=128, n_actions=4, n_hidden_layers=2).to(self.device)
        self.target_net = QNet(n_features=8, hidden_d=128, n_actions=4, n_hidden_layers=2).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.training_error = []

    def get_action_test(self, obs: np.ndarray) -> int:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action_values_tensor = self.q_net(obs_tensor)
        action = int(torch.argmax(action_values_tensor, dim=-1).item())
        return action

    def get_actions_boltzmann(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_values_tensor = self.q_net(obs_tensor)
        actions_dist = Categorical(logits=actions_values_tensor)
        actions_tensor = actions_dist.sample()
        actions = actions_tensor.detach().cpu().numpy()
        return actions

    def get_actions_epsilon(self, obs: np.ndarray, envs: gym.vector.SyncVectorEnv, epsilon: float) -> np.ndarray:
        n_envs = obs.shape[0]
        
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions_value_tensor = self.q_net(obs_tensor)
        actions = torch.argmax(actions_value_tensor, dim=-1).cpu().numpy()
        
        random_mask = np.random.rand(n_envs) < epsilon

        if np.any(random_mask):
            random_actions = envs.action_space.sample()
            actions[random_mask] = random_actions[random_mask]
        
        return actions
        
    def update_q_value(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        done: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.int, device=self.device)
        future_q_value = torch.as_tensor((1.0 - done) * self.discount_factor, dtype=torch.float32, device=self.device) * (torch.max(self.target_net(next_obs_tensor).detach(), -1).values)
        target = torch.as_tensor(rewards, dtype=torch.float32, device=self.device) + future_q_value

        loss = F.mse_loss(self.q_net(obs_tensor).gather(dim=1, index=rearrange(actions_tensor, "b -> b 1")).squeeze(-1), torch.as_tensor(target))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_error.append(loss.detach().cpu().numpy())
    
    def update_target_hard(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def update_target_soft(self, tau: float) -> None:
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, checkpoint_path: Union[str, Path]) -> Path:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "agent_hparams": {
                "device": self.device,
                "learning_rate": self.lr,
                "discount_factor": self.discount_factor,
                "epsilon": self.epsilon,
                "min_epsilon": self.min_epsilon,
                "decay_steps": self.decay_steps,
            },
            "qnet_hparams": {
                "n_features": self.q_net.n_features,
                "n_actions": self.q_net.n_actions,
                "hidden_d": self.q_net.hidden_d,
                "n_hidden_layers": self.q_net.n_hidden_layers
            },
            "q_net_state_dict": self.q_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_error": self.training_error,
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    @classmethod
    def load(cls, device: torch.device, env: gym.Env, checkpoint_path: Union[str, Path], map_location: str = "cpu"):
        checkpoint_path = Path(checkpoint_path)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

        agent_hparams = checkpoint["agent_hparams"]
        agent = cls(
            device=device,
            learning_rate=agent_hparams["learning_rate"],
            epsilon=agent_hparams["epsilon"],
            min_epsilon=agent_hparams["min_epsilon"],
            decay_steps=agent_hparams["decay_steps"],
            discount_factor=agent_hparams["discount_factor"],
        )

        agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        agent.q_net.eval()

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
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
    
    config = {
        "n_envs": 10,
        "learning_rate": 5e-4,
        "n_update_steps": 50000,
        "batch_size": 128,
        "buffer_size": 1000,
        "discount_factor": 0.99,
        "epsilon": 1.0,
        "min_epsilon": 0.05,
        "decay_steps": 25000,
        "sample_method": "boltzmann",
        "target_update_method": "soft",
        "target_update_step": 50,
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config["device"] = device

    n_envs = config["n_envs"]
    learning_rate = config["learning_rate"]
    n_update_steps = config["n_update_steps"]
    batch_size = config["batch_size"] 
    buffer_size = config["buffer_size"]
    discount_factor = config["discount_factor"]
    epsilon = config["epsilon"]
    min_epsilon = config["min_epsilon"]
    decay_steps = config["decay_steps"]
    sample_method = config["sample_method"]
    target_update_method = config["target_update_method"]
    if config["target_update_method"] == "hard":
        target_update_step = config["target_update_step"]
    else:
        target_update_tau = config["target_update_tau"]
    checkpoint_dir = Path("checkpoints")
    checkpoint_path = checkpoint_dir / f"{config['sample_method']}_{config['n_update_steps']}.pt"

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
            for i in range(10)
        ]
    )
    envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    agent = LunarLanderAgent(learning_rate=learning_rate, device=torch.device("mps"), discount_factor=discount_factor, epsilon=epsilon, min_epsilon=min_epsilon, decay_steps=decay_steps)
    replaybuffer = ReplayBuffer(buffer_size=buffer_size, observation_dim=8, action_dim=1, n_envs=10)
    
    obs, infos = envs_wrapper.reset() 

    for _ in tqdm(range(buffer_size)):
        action = agent.get_actions_epsilon(obs=obs, envs=envs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, infos = envs_wrapper.step(action)
        done = np.array([ter or tru for ter, tru in zip(terminated, truncated)])
        replaybuffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        obs = next_obs
    with logging_redirect_tqdm():
        for step in tqdm(range(n_update_steps)):
            cur_epsilon = epsilon - (epsilon - min_epsilon) * (step / decay_steps) if step <= decay_steps else min_epsilon
            data_dict = replaybuffer.sample(batch_size=batch_size)
            agent.update_q_value(obs=data_dict["obs"], actions=data_dict["actions"], rewards=data_dict["rewards"], next_obs=data_dict["next_obs"], done=data_dict["done"])
            if target_update_method == "soft":
                agent.update_target_soft(target_update_tau)
            actions = agent.get_actions_epsilon(obs=obs, envs=envs, epsilon=cur_epsilon)
            next_obs, rewards, terminated, truncated, infos = envs_wrapper.step(actions=actions)
            done = np.array([ter or tru for ter, tru in zip(terminated, truncated)])
            replaybuffer.add(obs=obs, action=actions, reward=rewards, next_obs=next_obs, done=done)
            obs = next_obs

            if "episode" in infos:
                for i in range(len(infos["_episode"])):
                    if infos["_episode"][i]:
                        episode_reward = infos["episode"]["r"][i]
                        episode_length = infos["episode"]["l"][i]

                        logger.info(f"episode_reward: {episode_reward}, episode_length: {episode_length}")
                        
                        wandb.log({
                            "train/reward": episode_reward,
                            "train/length": episode_length,
                            "train/epsilon": cur_epsilon,
                        })

    saved_path = agent.save(checkpoint_path)
    logger.info(f"Checkpoint saved to: {saved_path}")

