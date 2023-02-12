import torch
from torch import nn
import math

EPSILON = 1e-7  # Small value to avoid divide by zero


class ActorModel(nn.Module):
    def __init__(self, body_part, body_part_properties, input_dim):
        super().__init__()
        self.body_part = body_part

        # в оригинале здесь сначала LinearEncoder, но я сначала упрощаю модель
        self.actor_body = nn.Sequential(
            nn.Linear(input_dim, body_part_properties.hidden_dim),
            nn.Tanh(),
            nn.Linear(body_part_properties.hidden_dim, body_part_properties.output_dim)
        )

        # TODO: пронаблюдать, что обучается
        self.log_sigma = nn.Parameter(
            torch.zeros(1, body_part_properties.output_dim, requires_grad=True)
        )

    def forward(self, input_data):
        mean = self.actor_body(input_data)

        std = torch.exp(self.log_sigma)
        rand_shift = torch.randn_like(mean) * std
        actions = mean + rand_shift

        return actions

    def forward_with_stat(self, input_data):
        mean = self.actor_body(input_data)

        std = torch.exp(self.log_sigma)
        rand_shift = torch.randn_like(mean) * std
        actions = mean + rand_shift

        log_prob, entropy = self.get_stats(actions, mean, std)

        return actions, log_prob, entropy

    def get_stats(self, actions, mean, std):
        var = std ** 2
        log_scale_var = torch.log(std + EPSILON)

        log_prob = -((actions - mean) ** 2) / (2 * var + EPSILON) - log_scale_var - math.log(math.sqrt(2 * math.pi))

        entropy = torch.mean(0.5 * torch.log(2 * math.pi * math.e * std ** 2 + EPSILON), dim=1, keepdim=True)
        #entropy = torch.sum(entropy, dim=1)

        return log_prob, entropy

    def evaluate_actions(self, input_data, actions):
        mean = self.actor_body(input_data)
        std = torch.exp(self.log_sigma)

        log_prob, entropy = self.get_stats(actions, mean, std)

        return log_prob, entropy
