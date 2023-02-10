import numpy as np
import torch


class OnPolicyBatchReplay:
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.batch = {}
        self.batch_len = 0
        self.reset()

    def reset(self):
        self.batch = {}
        self.batch['states'] = []
        self.batch['actions'] = []
        self.batch['rewards'] = []
        self.batch['next_states'] = []
        self.batch['dones'] = []
        self.batch_len = 0

    def add_experience(self, state, action, reward, next_state, done):
        self.batch['states'].append(state)
        self.batch['actions'].append(action)
        self.batch['rewards'].append(reward)
        self.batch['next_states'].append(next_state)
        self.batch['dones'].append(done)
        self.batch_len += 1

    def sample(self):
        batch = {}
        for k in self.batch:
            bi = self.batch[k][:self.batch_size]
            npa = np.array(bi)
            ta = torch.from_numpy(npa.astype(np.float32)).to(self.device)
            batch[k] = ta
        self.reset()
        return batch

    def batch_is_full(self):
        return (self.batch_len >= self.batch_size)

