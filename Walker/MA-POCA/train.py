from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

import numpy as np
import torch
import random

from body_parts import WalkerBody
from body_parts import BodyPartProperties

from actor import ActorModel
from critic import CriticModel
from loss_functions import calc_value_loss

from memory import ExperienceBuffer

RANDOM_SEED = 42


def train():

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    walker_body = WalkerBody()

    params = list()
    body_model = dict()
    for body_part, body_part_property in walker_body.body.items():
        actor_model = ActorModel(body_part, body_part_property, walker_body.obs_size).to(device)
        params.extend(list(actor_model.parameters()))
        body_model[body_part] = actor_model

    critic_model = CriticModel(walker_body).to(device)
    params.extend(list(critic_model.parameters()))

    optimizer = torch.optim.Adam(params, lr=0.0003)

    env = UnityEnvironment("./Unity/Walker", worker_id=1, no_graphics=True)
    #env = UnityEnvironment("/home/tnv/tempWalker/Walker", no_graphics=False)
    env.reset()
    behavior_name = None
    for behavior_name in env.behavior_specs:
        print(behavior_name)
    action_spec = env.behavior_specs[behavior_name].action_spec
    observation_specs = env.behavior_specs[behavior_name].observation_specs

    memory = None
    buffer_size = 32

    while True:
        ds, ts = env.get_steps(behavior_name)
        if ds.agent_id.shape[0] > 0:
            assert len(ds.obs) == 1, f"len(ds.obs) != 1"

            if memory is None:
                memory = ExperienceBuffer(walker_body, ds.agent_id, buffer_size)
            n_agents = ds.agent_id.shape[0]

            continious_actions = torch.zeros(n_agents, action_spec.continuous_size)
            input_data = torch.from_numpy(ds.obs[0])
            log_probs = torch.zeros(n_agents, action_spec.continuous_size)
            entropies = dict()
            for body_part, body_part_model in body_model.items():
                with torch.no_grad():
                    actions, log_prob, entropy = body_part_model.forward_with_stat(input_data)
                idxs = walker_body.body[body_part].action_space_idxs
                continious_actions[:, idxs] = actions
                log_probs[:, idxs] = log_prob
                entropies[body_part] = entropy

            action_tuple = ActionTuple()
            action_tuple.add_continuous(continious_actions.detach().cpu().numpy())
            #random_actions = action_spec.random_action(n_agents)
            env.set_actions(behavior_name, action_tuple)

            for agent_id in ds.agent_id:
                idx = ds.agent_id_to_index[agent_id]
                memory.add_experience(agent_id, torch.from_numpy(ds.obs[0][idx]), continious_actions[idx], ds.reward[idx], False,
                                      log_probs[idx], entropies)

        if ts.agent_id.shape[0] > 0:
            # TODO: нужно реализовать завершение эпизода для агента
            n_agents = ds.agent_id.shape[0]

        env.step()

        if memory.batch_is_full():
            for agent_id in memory.agent_ids:
                critic_batch = memory.get_last_data(agent_id)
                for body_part in walker_body.body.keys():
                    values_full = critic_model.critic_full(critic_batch)
                    values_body_part = critic_model.critic_body_part(critic_batch, body_part)
                    memory.update_last_data(agent_id, body_part, values_full, values_body_part)

            for agent_id in memory.agent_ids:
                batch = memory.sample(agent_id)
                body_part_names = list(walker_body.body.keys())
                random.shuffle(body_part_names)
                for body_part in body_part_names:
                    obs = batch[body_part]["obs"]
                    actions = batch[body_part]["actions"]
                    log_prob, entropy = body_model[body_part].evaluate_actions(obs, actions)

                    values_full = critic_model.critic_full(batch)

                    values_body_part = critic_model.critic_body_part(batch, body_part)

                    value_body_loss = calc_value_loss(values_full, old_body_values, returns)
                    value_loss = calc_value_loss(values_full, old_values, returns)

            memory.reset()



if __name__ == '__main__':
    train()

