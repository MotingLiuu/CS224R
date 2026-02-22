from collections import defaultdict
from datetime import datetime
from pathlib import Path
import gymnasium as gym
import numpy as np
import pickle

class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate : float,
        initial_epsilon : float,
        epsilon_decay : float,
        final_epsilon : float,
        discount_factor : float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        
        self.lr = learning_rate
        self.discount_factor = discount_factor
        
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        self.training_error = []

    def get_action_epsilon(self, obs: tuple[int, int, bool]) -> int:

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore: random action
        else:
            return np.argmax(self.q_values[obs])  # Exploit: best known action
        
    def get_action_boltzmann(self, obs: tuple[int, int, bool]) -> int:
        q_values = self.q_values[obs]
        exp_q = np.exp(q_values - np.max(q_values))  
        action_probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=action_probs)
        
    def get_action_test(self, obs: tuple[int, int, bool]) -> int:
        
        return np.argmax(self.q_values[obs]) 
    
    def get_random_action(self, obs: tuple[int, int, bool]) -> int:
        return self.env.action_space.sample()

    def update_q_values(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool],
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value

        temporal_difference = target - self.q_values[obs][action]

        self.q_values[obs][action] += self.lr * temporal_difference

        self.training_error.append(abs(temporal_difference))

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_q_table(self, file_path: str | Path):
        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "q_values": {state: values.tolist() for state, values in self.q_values.items()},
            "hyperparameters": {
                "learning_rate": self.lr,
                "initial_epsilon": self.initial_epsilon,
                "epsilon_decay": self.epsilon_decay,
                "final_epsilon": self.final_epsilon,
                "discount_factor": self.discount_factor,
            },
        }

        with target_path.open("wb") as f:
            pickle.dump(payload, f)

    def load_q_table(self, file_path: str | Path):
        source_path = Path(file_path)
        with source_path.open("rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict) and "q_values" in payload:
            q_values = payload["q_values"]
        else:
            q_values = payload

        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        for state, values in q_values.items():
            self.q_values[state] = np.array(values, dtype=np.float32)

        if isinstance(payload, dict) and "hyperparameters" in payload:
            return payload["hyperparameters"]
        return None

    @classmethod
    def from_saved_file(cls, env: gym.Env, file_path: str | Path):
        source_path = Path(file_path)
        with source_path.open("rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict) or "hyperparameters" not in payload:
            raise ValueError("Saved file does not contain hyperparameters.")

        hyperparameters = payload["hyperparameters"]
        agent = cls(
            env=env,
            learning_rate=hyperparameters["learning_rate"],
            initial_epsilon=hyperparameters["initial_epsilon"],
            epsilon_decay=hyperparameters["epsilon_decay"],
            final_epsilon=hyperparameters["final_epsilon"],
            discount_factor=hyperparameters["discount_factor"],
        )
        agent.load_q_table(file_path)
        return agent

if __name__ == "__main__":

    import wandb

    config = {
        "learning_rate": 0.01,
        "n_episodes": 100_000,
        "start_epsilon": 1.0, 
        "final_epsilon": 0.1,
        "discount_factor": 0.95,
        "boltzmann": False,
    }

    epsilon_decay = (config["start_epsilon"]) / (config["n_episodes"] * 0.5)

    if config["boltzmann"]:
        run_name = datetime.now().strftime("blackjack_boltzmann_%Y%m%d_%H%M%S")
    else:
        run_name = datetime.now().strftime("blackjack_epsilon_greedy_%Y%m%d_%H%M%S")
    wandb.init(
        project="gymrl",
        name=run_name,
        config=config,
        )

    env = gym.make("Blackjack-v1")
    agent = BlackjackAgent(
        env=env,
        learning_rate=config["learning_rate"],
        initial_epsilon=config["start_epsilon"],
        epsilon_decay=epsilon_decay,
        final_epsilon=config["final_epsilon"],
    )

    for episode in range(config["n_episodes"]):
        obs, info = env.reset()
        episode_over = False
        episode_reward = 0.0
        td_error_start_idx = len(agent.training_error)

        while not episode_over:
            action = agent.get_action_epsilon(obs) if not config["boltzmann"] else agent.get_action_boltzmann(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.update_q_values(obs, action, reward, terminated, next_obs)

            obs = next_obs
            episode_over = terminated or truncated
            episode_reward += reward

        episode_td_errors = agent.training_error[td_error_start_idx:]
        mean_td_error = float(np.mean(episode_td_errors)) if episode_td_errors else 0.0

        wandb.log({
            "train/episode_reward": episode_reward,
            "train/epsilon": agent.epsilon,
            "train/mean_td_error": mean_td_error,
            "episode": episode,
        })

        agent.decay_epsilon()

    project_root = Path(__file__).resolve().parents[1]
    run_name = wandb.run.name if wandb.run is not None else datetime.now().strftime("blackjack_%Y%m%d_%H%M%S")
    q_table_path = project_root / "data" / f"{run_name}_q_table.pkl"
    agent.save_q_table(q_table_path)

    reloaded_agent = BlackjackAgent.from_saved_file(env=env, file_path=q_table_path)

    sample_state = (20, 10, False)
    print(f"Saved Q-table to: {q_table_path}")
    print(f"Reloaded Q-values for {sample_state}: {reloaded_agent.q_values[sample_state]}")

    wandb.finish()
    env.close()
  

