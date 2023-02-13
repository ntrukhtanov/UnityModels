import torch
from torch import nn

from multi_agent_network_body import MultiAgentNetworkBody

HIDDEN_SIZE = 128


class CriticModel(nn.Module):
    def __init__(self, walker_body):
        super().__init__()
        self.multiagent_network = MultiAgentNetworkBody(walker_body, HIDDEN_SIZE)

        value_encoding_size = HIDDEN_SIZE #  TODO: + 1 для количества агентов, на первом этапе руки роботу не ломаем
        self.value_heads = nn.Linear(value_encoding_size, 1)

    def forward(self, encoding):
        return self.value_heads(encoding)

    def critic_full(self, batch):
        encoding = self.multiagent_network(batch=batch, body_part=None)

        value_outputs = self.forward(encoding)

        return value_outputs

    def critic_body_part(self, batch, body_part):
        encoding = self.multiagent_network(batch=batch, body_part=body_part)

        value_outputs = self.forward(encoding)

        return value_outputs
