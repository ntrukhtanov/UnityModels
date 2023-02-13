import torch
from torch import nn


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_size):
        super().__init__()

        layers = list()
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.Sigmoid())
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*layers)

    def forward(self, inp):
        print(f"LinearEncoder forward inp is in cuda {inp.is_cuda}")
        for p in self.encoder.parameters():
            print(f"LinearEncoder parameter is in cuda {p.is_cuda}")
        return self.encoder(inp)
