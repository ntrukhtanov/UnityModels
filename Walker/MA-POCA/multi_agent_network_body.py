import torch
from torch import nn

from linear_encoder import LinearEncoder
from entity_embedding import EntityEmbedding
from attention import ResidualSelfAttention

EMBEDDING_SIZE = 128  # в оригинале 256


class MultiAgentNetworkBody(nn.Module):
    def __init__(self, walker_body, hidden_size):
        super().__init__()

        self.walker_body = walker_body

        self.obs_encoders = dict()
        self.obs_action_encoders = dict()

        for body_part, body_part_prop in walker_body.body.items():
            self.obs_encoders[body_part] = EntityEmbedding(body_part_prop.input_dim, EMBEDDING_SIZE)
            self.obs_action_encoders[body_part] = EntityEmbedding(
                body_part_prop.input_dim + body_part_prop.output_dim, EMBEDDING_SIZE)

        self.self_attn = ResidualSelfAttention(EMBEDDING_SIZE)

        self.linear_encoder = LinearEncoder(EMBEDDING_SIZE, 2, hidden_size)

    def forward(self, batch, body_part):
        self_attn_inputs = list()

        if body_part is None:
            all_encoded_obs = list()
            for body_part in self.walker_body.body.keys():
                all_encoded_obs.append(self.obs_encoders[body_part](batch[body_part]["obs"]))
            all_encoded_obs = torch.stack(all_encoded_obs, dim=1)
            self_attn_inputs.append(all_encoded_obs)
        else:
            encoded_obs = self.obs_encoders[body_part](batch[body_part]["obs"])
            encoded_obs = torch.stack([encoded_obs], dim=1)
            self_attn_inputs.append(encoded_obs)

            encoded_obs_with_actions = list()
            for other_body_part in self.walker_body.body.keys():
                if other_body_part != body_part:
                    obs = batch[other_body_part]["obs"]
                    actions = batch[other_body_part]["actions"]
                    obs_with_actions = torch.cat([obs, actions], dim=1)
                    encoded_obs_with_actions.append(self.obs_action_encoders[other_body_part](obs_with_actions))
            encoded_obs_with_actions = torch.stack(encoded_obs_with_actions, dim=1)
            self_attn_inputs.append(encoded_obs_with_actions)

        encoded_entity = torch.cat(self_attn_inputs, dim=1)
        encoded_state = self.self_attn(encoded_entity)

        # TODO: здесь нужно добавить количество агентов, на первом этапе не делаем

        encoding = self.linear_encoder(encoded_state)

        return encoding
