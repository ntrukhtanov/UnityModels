from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

import numpy as np
import torch
import random

from body_parts import WalkerBody
from body_parts import BodyPartProperties

from actor import ActorModel
from critic import CriticModel


RANDOM_SEED = 42


def train():

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    walker_body = WalkerBody()

    body_model = list()
    for body_part, body_part_property in walker_body.body.items():
        body_model.append(ActorModel(body_part, body_part_property))

    critic_model = CriticModel()

    env = UnityEnvironment("./Unity/Walker", no_graphics=False)
    #env = UnityEnvironment("/home/tnv/tempWalker/Walker", no_graphics=False)
    env.reset()
    behavior_name = None
    for behavior_name in env.behavior_specs:
        print(behavior_name)
    action_spec = env.behavior_specs[behavior_name].action_spec
    observation_specs = env.behavior_specs[behavior_name].observation_specs
    while True:
        ds, ts = env.get_steps(behavior_name)
        if ds.agent_id.shape[0] > 0:
            n_agents = ds.agent_id.shape[0]

            continious_actions = torch.zeros(n_agents, action_spec.continuous_size)
            input_data = torch.from_numpy(ds.obs[0])
            for body_part_model in body_model:
                actions, log_prob, entropy = body_part_model.forward_with_stat(input_data)
                idxs = walker_body.body[body_part_model.body_part].action_space_idxs
                continious_actions[:, idxs] = actions

            action_tuple = ActionTuple()
            action_tuple.add_continuous(continious_actions.detach().cpu().numpy())

            #random_actions = action_spec.random_action(n_agents)
            env.set_actions(behavior_name, action_tuple)

        if ts.agent_id.shape[0] > 0:
            # TODO: нужно реализовать завершение эпизода для агента
            n_agents = ds.agent_id.shape[0]

        if True: # TODO: здесь условие запуска процесса обучения на накопленном батче
            pass

        env.step()


if __name__ == '__main__':
    train()

