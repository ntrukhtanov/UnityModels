from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

import numpy as np
import torch
import random
from torch.utils.tensorboard import SummaryWriter
import sys
from tqdm import tqdm
import statistics

from body_parts import WalkerBody
from body_parts import BodyPartProperties

from actor import ActorModel
from critic import CriticModel
from loss_functions import calc_value_loss
from loss_functions import calc_returns
from loss_functions import calc_policy_loss

from memory import ExperienceBuffer

RANDOM_SEED = 42

GAMMA = 0.99
LAM = 0.95
EPSILON = 0.2
BETA = 0.01


def train(model_file, summary_dir, total_steps, buffer_size):
    assert model_file is not None, f"Не указан обязательный параметр model_file"
    assert summary_dir is not None, f"Не указан обязательный параметр summary_dir"
    assert total_steps is not None, f"Не указан обязательный параметр total_steps"
    assert buffer_size is not None, f"Не указан обязательный параметр buffer_size"

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    summary_writer = SummaryWriter(summary_dir)

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

    env = UnityEnvironment(model_file, worker_id=1, no_graphics=True)
    #env = UnityEnvironment("/home/tnv/tempWalker/Walker", no_graphics=False)
    env.reset()
    behavior_name = None
    for behavior_name in env.behavior_specs:
        print(behavior_name)
    action_spec = env.behavior_specs[behavior_name].action_spec
    observation_specs = env.behavior_specs[behavior_name].observation_specs

    memory = None

    agents_summary_rewards = dict()

    for step in tqdm(range(total_steps)):
        ds, ts = env.get_steps(behavior_name)
        if ds.agent_id.shape[0] > 0:
            assert len(ds.obs) == 1, f"len(ds.obs) != 1"

            if memory is None:
                memory = ExperienceBuffer(walker_body, ds.agent_id, buffer_size)
            n_agents = ds.agent_id.shape[0]

            continious_actions = torch.zeros(n_agents, action_spec.continuous_size)
            input_data = torch.from_numpy(ds.obs[0])
            log_probs = torch.zeros(n_agents, action_spec.continuous_size)
            for body_part, body_part_model in body_model.items():
                with torch.no_grad():
                    input_data = input_data.to(device)
                    actions, log_prob, entropy = body_part_model.forward_with_stat(input_data)
                idxs = walker_body.body[body_part].action_space_idxs
                continious_actions[:, idxs] = actions.cpu().detach()
                log_probs[:, idxs] = log_prob.cpu().detach()

            action_tuple = ActionTuple()
            action_tuple.add_continuous(continious_actions.detach().cpu().numpy())
            #random_actions = action_spec.random_action(n_agents)
            env.set_actions(behavior_name, action_tuple)

            for agent_id in ds.agent_id:
                idx = ds.agent_id_to_index[agent_id]
                memory.add_experience(agent_id, torch.from_numpy(ds.obs[0][idx]), continious_actions[idx],
                                      ds.reward[idx], False, log_probs[idx])
                summary_reward = agents_summary_rewards.get(agent_id, 0.0)
                summary_reward += ds.reward[idx]
                agents_summary_rewards[agent_id] = summary_reward

        if ts.agent_id.shape[0] > 0:
            summary_rewards = list()
            for agent_id in ts.agent_id:
                idx = ts.agent_id_to_index[agent_id]
                obs = torch.from_numpy(ts.obs[0][idx])
                reward = ts.reward[idx]
                memory.set_terminate_state(agent_id, obs, reward)

                summary_rewards.append(agents_summary_rewards[agent_id] + reward)
                agents_summary_rewards[agent_id] = 0.0

            summary_writer.add_scalar("Total reward", statistics.mean(summary_rewards), step)

        env.step()

        if memory.batch_is_full():
            estimates = dict()
            for agent_id in memory.agent_ids:
                estimates[agent_id] = dict()
                batch = memory.sample(agent_id, device)
                last_data = memory.get_last_data(agent_id, device)
                for body_part in walker_body.body.keys():
                    with torch.no_grad():
                        estimates[agent_id][body_part] = dict()
                        old_values_full = critic_model.critic_full(batch)
                        estimates[agent_id][body_part]["old_values_full"] = old_values_full.detach()
                        old_values_body_part = critic_model.critic_body_part(batch, body_part)
                        estimates[agent_id][body_part]["old_values_body_part"] = old_values_body_part.detach()

                        values_next = critic_model.critic_full(last_data)
                        estimates[agent_id][body_part]["values_next"] = values_next.detach()

                        returns = calc_returns(batch[body_part]["rewards"], batch[body_part]["dones"], old_values_full,
                                               GAMMA,
                                               LAM, values_next)
                        estimates[agent_id][body_part]["returns"] = returns.detach()

                        advantages = returns.unsqueeze(1) - old_values_body_part
                        estimates[agent_id][body_part]["advantages"] = advantages.detach()

            for agent_id in memory.agent_ids:
                batch = memory.sample(agent_id, device)
                body_part_names = list(walker_body.body.keys())
                random.shuffle(body_part_names)
                for body_part in body_part_names:
                    obs = batch[body_part]["obs"]
                    actions = batch[body_part]["actions"]
                    log_probs, entropy = body_model[body_part].evaluate_actions(obs, actions)

                    values_full = critic_model.critic_full(batch)

                    values_body_part = critic_model.critic_body_part(batch, body_part)

                    old_values_full = estimates[agent_id][body_part]["old_values_full"]
                    old_values_body_part = estimates[agent_id][body_part]["old_values_body_part"]
                    returns = estimates[agent_id][body_part]["returns"]

                    value_body_loss = calc_value_loss(values_body_part, old_values_body_part, returns, EPSILON)
                    value_loss = calc_value_loss(values_full, old_values_full, returns, EPSILON)

                    old_log_probs = batch[body_part]["log_probs"]
                    advantages = estimates[agent_id][body_part]["advantages"]

                    policy_loss = calc_policy_loss(advantages, log_probs, old_log_probs, EPSILON)

                    loss = (
                            policy_loss
                            + 0.5 * (value_loss + 0.5 * value_body_loss)
                            - BETA * torch.mean(entropy)
                    )

                    summary_writer.add_scalar("loss", loss.item(), step)
                    summary_writer.add_scalar("policy_loss", policy_loss.item(), step)
                    summary_writer.add_scalar("value_loss", value_loss.item(), step)
                    summary_writer.add_scalar("value_body_loss", value_body_loss.item(), step)
                    summary_writer.add_scalar("entropy", torch.mean(entropy).item(), step)

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

            memory.reset()


def run():
    args = sys.argv

    if '-summary_dir' in args:
        idx = args.index('-summary_dir')
        summary_dir = args[idx + 1]
    else:
        summary_dir = None

    if '-total_steps' in args:
        idx = args.index('-total_steps')
        total_steps = int(args[idx + 1])
    else:
        total_steps = None

    if '-buffer_size' in args:
        idx = args.index('-buffer_size')
        buffer_size = int(args[idx + 1])
    else:
        buffer_size = None

    if '-model_file' in args:
        idx = args.index('-model_file')
        model_file = args[idx + 1]
    else:
        model_file = None

    train(model_file=model_file,
          summary_dir=summary_dir,
          total_steps=total_steps,
          buffer_size=buffer_size)


if __name__ == '__main__':
    run()

