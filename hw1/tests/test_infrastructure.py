from cs224r.infrastructure.pytorch_util import build_mlp
from cs224r.infrastructure.pytorch_util import MLP
from cs224r.infrastructure.replay_buffer import ReplayBuffer
import torch
import torch.nn as nn
import logging
import numpy as np

logger = logging.getLogger(__name__)

class TestUtil:
    
    def test_build_mlp(self):
        input_size = 4
        output_size = 2
        n_layers = 3
        size = 8
        activation = 'relu'
        output_activation = 'tanh'
        
        mlp = build_mlp(input_size, output_size, n_layers, size, activation, output_activation)
        
        assert isinstance(mlp, nn.Module), "build_mlp should return an instance of nn.Module"
        
        x = torch.randn(5, input_size)
        output = mlp(x)
        
        assert output.shape == (5, output_size), f"Expected output shape (5, {output_size}), but got {output.shape}"

        logger.debug("MLP structure:\n%s", mlp)

    def test_sample_random_data(self):

        buffer = ReplayBuffer(max_size=100)
        buffer.add_rollouts(
            paths=[
                {
                    "observation": np.array([1, 2]),
                    "action": np.array([0, 1]),
                    "reward": np.array([1, 1]),
                    "next_observation": np.array([3, 4]),
                    "terminal": np.array([0, 1]),
                },
                {
                    "observation": np.array([5, 6]),
                    "action": np.array([0, 1]),
                    "reward": np.array([1, 1]),
                    "next_observation": np.array([7, 8]),
                    "terminal": np.array([0, 1]),
                }
            ]
        )
        
        sample_size = 1
        obs, acs, rews, next_obs, terminals = buffer.sample_random_data(batch_size=sample_size)
        
        logger.debug("Sampled observations: %s", obs)
        logger.debug("Sampled actions: %s", acs)
        logger.debug("Sampled rewards: %s", rews)
        logger.debug("Sampled next observations: %s", next_obs)
        logger.debug("Sampled terminals: %s", terminals)



