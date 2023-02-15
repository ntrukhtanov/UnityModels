import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class ExperienceBuffer:
    def __init__(self, walker_body, agent_ids, buffer_size):
        self.walker_body = walker_body
        self.agent_ids = agent_ids
        self.buffer_size = buffer_size

        self.buffer = dict()

        self.reset()

    def reset(self):
        self.buffer = dict()
        for agent_id in self.agent_ids:
            self.buffer[agent_id] = dict()
            self.buffer[agent_id]["obs"] = list()
            self.buffer[agent_id]["actions"] = list()
            self.buffer[agent_id]["log_probs"] = list()
            self.buffer[agent_id]["rewards"] = list()
            self.buffer[agent_id]['next_obs'] = list()
            self.buffer[agent_id]['dones'] = list()

    def add_experience(self, agent_id, obs, actions, reward, done, log_probs):
        if len(self.buffer[agent_id]["obs"]) > len(self.buffer[agent_id]["next_obs"]):
            self.buffer[agent_id]["next_obs"].append(obs)
        self.buffer[agent_id]["obs"].append(obs)
        self.buffer[agent_id]["actions"].append(actions)
        self.buffer[agent_id]["rewards"].append(reward)
        self.buffer[agent_id]["dones"].append(done)
        self.buffer[agent_id]["log_probs"].append(log_probs)

    def set_terminate_state(self, agent_id, obs, reward):
        if len(self.buffer[agent_id]["rewards"]) > 0:
            self.buffer[agent_id]["next_obs"].append(obs)
            last_idx = len(self.buffer[agent_id]["rewards"]) - 1
            self.buffer[agent_id]["rewards"][last_idx] = reward
            self.buffer[agent_id]["dones"][last_idx] = True

    def add_calculated_values(self, agent_id, body_part, key, data):
        if body_part not in self.buffer[agent_id].keys():
            self.buffer[agent_id][body_part] = dict()
        self.buffer[agent_id][body_part][key] = data

    def create_train_data_loaders(self):
        for body_part, body_part_prop in self.walker_body.body.items():

    def sample(self, agent_id, device):
        buffer = dict()
        for body_part, body_part_prop in self.walker_body.body.items():
            buffer[body_part] = dict()
            buffer[body_part]["full_obs"] = torch.stack(self.buffer[agent_id]["obs"][:self.buffer_size], dim=0).to(
                device)
            buffer[body_part]["obs"] = torch.stack(self.buffer[agent_id]["obs"][:self.buffer_size], dim=0)[:,
                                       body_part_prop.obs_space_idxs].to(
                device)
            buffer[body_part]["actions"] = torch.stack(
                self.buffer[agent_id]["actions"][:self.buffer_size], dim=0)[:, body_part_prop.action_space_idxs].to(
                device)
            buffer[body_part]["rewards"] = torch.Tensor(self.buffer[agent_id]["rewards"][:self.buffer_size]).to(
                device)
            buffer[body_part]["dones"] = torch.Tensor(self.buffer[agent_id]["dones"][:self.buffer_size]).to(device)
            buffer[body_part]["log_probs"] = torch.stack(
                self.buffer[agent_id]["log_probs"][:self.buffer_size], dim=0)[:,body_part_prop.action_space_idxs].to(
                device)
        return buffer

    def get_last_data(self, agent_id, device):
        buffer = dict()
        for body_part, body_part_prop in self.walker_body.body.items():
            buffer[body_part] = dict()
            if self.buffer[agent_id]["dones"][self.buffer_size - 1]:
                buffer[body_part]["obs"] = torch.stack([self.buffer[agent_id]["next_obs"][self.buffer_size - 1]],
                                                       dim=0)[:, body_part_prop.obs_space_idxs].to(device)
            else:
                buffer[body_part]["obs"] = torch.stack([self.buffer[agent_id]["obs"][self.buffer_size]], dim=0)[:,
                                           body_part_prop.obs_space_idxs].to(device)
        return buffer

    def batch_is_full(self):
        for agent_id in self.agent_ids:
            if len(self.buffer[agent_id]["actions"]) < self.buffer_size + 1:
                return False
        return True

