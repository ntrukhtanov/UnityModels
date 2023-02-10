from mlagents_envs.environment import UnityEnvironment
from tqdm import tqdm

import random
import numpy as np
import torch

from DungeonEscape.PPO.on_policy_batch_replay import OnPolicyBatchReplay


RANDOM_SEED = 7


def train(env_file_name,
          no_graphics,
          total_steps_count):

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    env = UnityEnvironment(env_file_name, no_graphics=no_graphics)
    env.reset()
    behavior_name = None
    for behavior_name in env.behavior_specs:
        print(behavior_name)

    for step in tqdm(range(total_steps_count)):
        ds, ts = env.get_steps(behavior_name)
        #if ts.agent_id.shape[0] > 0:


if __name__ == '__main__':
    env_file_name = './Unity/DungeonEscape'
    no_graphics = False
    total_steps_count = 20000000
    train(env_file_name,
          no_graphics,
          total_steps_count)

