import random

import torch

# TODO: прокомментировать
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

            self.buffer[agent_id]["common_values"] = list()
            for body_part in self.walker_body.body.keys():
                self.buffer[agent_id][body_part] = dict()
                self.buffer[agent_id][body_part]["body_part_values"] = list()

    def add_experience(self, agent_id, obs, actions, reward, log_probs):
        if len(self.buffer[agent_id]["obs"]) > len(self.buffer[agent_id]["next_obs"]):
            self.buffer[agent_id]["next_obs"].append(obs)
            self.buffer[agent_id]["rewards"].append(reward)
            self.buffer[agent_id]["dones"].append(False)
        self.buffer[agent_id]["obs"].append(obs)
        self.buffer[agent_id]["actions"].append(actions)
        self.buffer[agent_id]["log_probs"].append(log_probs)

    def set_terminate_state(self, agent_id, obs, reward):
        if len(self.buffer[agent_id]["obs"]) > 0:
            self.buffer[agent_id]["next_obs"].append(obs)
            self.buffer[agent_id]["rewards"].append(reward)
            self.buffer[agent_id]["dones"].append(True)

    def add_common_values(self, values):
        for i, agent_id in enumerate(self.agent_ids):
            self.buffer[agent_id]["common_values"].append(values[i])

    def add_body_part_values(self, body_part, values):
        for i, agent_id in enumerate(self.agent_ids):
            self.buffer[agent_id][body_part]["body_part_values"].append(values[i])

    def add_returns(self, agent_id, returns):
        self.buffer[agent_id]["returns"] = returns

    def add_advantages(self, agent_id, body_part, advantages):
        self.buffer[agent_id][body_part]["advantages"] = advantages

    def add_calculated_values(self, agent_id, body_part, key, data):
        if body_part not in self.buffer[agent_id].keys():
            self.buffer[agent_id][body_part] = dict()
        self.buffer[agent_id][body_part][key] = data

    def sample_buffer(self, shuffle):
        buffer = dict()
        buffer['size'] = self.buffer_size * len(self.agent_ids)
        rnd_idxs = list(range(buffer['size']))
        if shuffle:
            random.shuffle(rnd_idxs)
        for body_part, body_part_prop in self.walker_body.body.items():
            buffer[body_part] = dict()
            buffer[body_part]["full_obs"] = list()
            buffer[body_part]["obs"] = list()
            buffer[body_part]["actions"] = list()
            buffer[body_part]["log_probs"] = list()
            buffer[body_part]["common_values"] = list()
            buffer[body_part]["body_part_values"] = list()
            buffer[body_part]["returns"] = list()
            buffer[body_part]["advantages"] = list()
            for agent_id in self.agent_ids:
                buffer[body_part]["full_obs"].append(torch.stack(self.buffer[agent_id]["obs"][:self.buffer_size], dim=0))

                buffer[body_part]["obs"].append(torch.stack(self.buffer[agent_id]["obs"][:self.buffer_size], dim=0)[:,
                                           body_part_prop.obs_space_idxs])

                buffer[body_part]["actions"].append(torch.stack(
                    self.buffer[agent_id]["actions"][:self.buffer_size], dim=0)[:, body_part_prop.action_space_idxs])

                buffer[body_part]["log_probs"].append(torch.stack(
                    self.buffer[agent_id]["log_probs"][:self.buffer_size], dim=0)[:, body_part_prop.action_space_idxs])

                buffer[body_part]["common_values"].append(
                    torch.stack(self.buffer[agent_id]["common_values"], dim=0)[:self.buffer_size])

                buffer[body_part]["body_part_values"].append(
                    torch.stack(self.buffer[agent_id][body_part]["body_part_values"], dim=0)[:self.buffer_size])

                buffer[body_part]["returns"].append(self.buffer[agent_id]["returns"])

                buffer[body_part]["advantages"].append(self.buffer[agent_id][body_part]["advantages"])

            buffer[body_part]["full_obs"] = torch.cat(buffer[body_part]["full_obs"], dim=0)[rnd_idxs]
            buffer[body_part]["obs"] = torch.cat(buffer[body_part]["obs"], dim=0)[rnd_idxs]
            buffer[body_part]["actions"] = torch.cat(buffer[body_part]["actions"], dim=0)[rnd_idxs]
            buffer[body_part]["log_probs"] = torch.cat(buffer[body_part]["log_probs"], dim=0)[rnd_idxs]
            buffer[body_part]["common_values"] = torch.cat(buffer[body_part]["common_values"], dim=0)[rnd_idxs]
            buffer[body_part]["body_part_values"] = torch.cat(buffer[body_part]["body_part_values"], dim=0)[rnd_idxs]
            buffer[body_part]["returns"] = torch.cat(buffer[body_part]["returns"], dim=0)[rnd_idxs]
            buffer[body_part]["advantages"] = torch.cat(buffer[body_part]["advantages"], dim=0)[rnd_idxs]

        return buffer

    def sample_batch(self, buffer, batch_step, batch_size, device, device_keys):
        batch = dict()
        for body_part in self.walker_body.body.keys():
            batch[body_part] = dict()
            for k in buffer[body_part].keys():
                batch[body_part][k] = buffer[body_part][k][batch_step:batch_step+batch_size]
                if k in device_keys:
                    batch[body_part][k] = batch[body_part][k].to(device)
        return batch

    def get_last_record_batch(self, device):
        batch = dict()
        for body_part, body_part_prop in self.walker_body.body.items():
            batch[body_part] = dict()
            batch[body_part]["obs"] = list()
            batch[body_part]["actions"] = list()
            for agent_id in self.agent_ids:
                batch[body_part]["obs"].append(self.buffer[agent_id]["obs"][-1])
                batch[body_part]["actions"].append(self.buffer[agent_id]["actions"][-1])

            batch[body_part]["obs"] = torch.stack(batch[body_part]["obs"], dim=0)[:,
                                      body_part_prop.obs_space_idxs].to(device)
            batch[body_part]["actions"] = torch.stack(batch[body_part]["actions"], dim=0)[:,
                                          body_part_prop.action_space_idxs].to(device)
        return batch

    def buffer_is_full(self):
        for agent_id in self.agent_ids:
            if len(self.buffer[agent_id]["actions"]) < self.buffer_size + 1:
                return False
        return True
