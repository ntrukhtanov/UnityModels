import torch
from torch import nn

from linear_encoder import LinearEncoder


class EntityEmbedding(nn.Module):
    def __init__(self, entity_size, embedding_size):
        """
        Всомогательная модель для эмбедингов
        :param entity_size: Размерность входных данный
        :param embedding_size: Размерность эмбедингов
        """
        super().__init__()

        self.embedding = LinearEncoder(entity_size, 1, embedding_size)

    def forward(self, inp):
        return self.embedding(inp)
