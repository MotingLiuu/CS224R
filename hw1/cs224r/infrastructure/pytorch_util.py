'''
TO EDIT: Utilities for handling PyTorch

Functions to edit:
    1. build_mlp (line 26) 
'''


from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

device = None

class MLP(nn.Module):
    def __init__(self, input_size, output_size, n_layers, size, activation=nn.Tanh(), output_activation=nn.Identity()):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = size
        self.output_size = output_size
        self.n_layers = n_layers
        self.activation = activation
        self.output_activation = output_activation

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.n_layers)])
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        
        return x

def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        Arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        Returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.

    return MLP(input_size, output_size, n_layers, size, activation, output_activation)


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("PyTorch detects an Apple GPU: running on MPS")
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
