from cs224r.infrastructure.pytorch_util import build_mlp
from cs224r.infrastructure.pytorch_util import MLP
import torch
import torch.nn as nn
import logging

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

