import torch
from torch import nn


class ResidualSelfAttention(torch.nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        # TODO: разобраться с MultiheadAttention и с параметром num_heads
        self.num_heads = 4

        self.attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=self.num_heads)

        self.fc_q = nn.Linear(embedding_size, embedding_size)
        self.fc_k = nn.Linear(embedding_size, embedding_size)
        self.fc_v = nn.Linear(embedding_size, embedding_size)
        self.fc_out = nn.Linear(embedding_size, embedding_size)

        self.embedding_norm = nn.LayerNorm(normalized_shape=embedding_size, elementwise_affine=False)
        self.residual_norm = nn.LayerNorm(normalized_shape=embedding_size, elementwise_affine=False)

    def forward(self, inp):
        inp = self.embedding_norm(inp)

        query = self.fc_q(inp)
        key = self.fc_k(inp)
        value = self.fc_v(inp)

        output, _ = self.attention(query, key, value)

        # Residual
        output = self.fc_out(output) + inp
        output = self.residual_norm(output)

        # Average Pooling
        #num_ent = inp.shape[1]
        #numerator = torch.sum(output * (1 - mask).reshape(-1, num_ent, 1), dim=1)
        #denominator = torch.sum(1 - mask, dim=1, keepdim=True) + self.EPSILON
        #output = numerator / denominator

        output = torch.mean(output, dim=1)

        return output
