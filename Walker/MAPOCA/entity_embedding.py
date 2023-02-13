import torch
from torch import nn

from linear_encoder import LinearEncoder


class EntityEmbedding(nn.Module):
    def __init__(self, entity_size, embedding_size):
        super().__init__()

        self.embedding = LinearEncoder(entity_size, 1, embedding_size)

    def forward(self, inp):
        return self.embedding(inp)
