from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import torch

from body_parts import WalkerBody
from actor import ActorModel

import sys
import time


# TODO: прокомментировать
# ./Unity/Walker
def run_model(walker_env_path, restore_path, env_worker_id):
    walker_body = WalkerBody()
    body_model = dict()

    for body_part, body_part_property in walker_body.body.items():
        actor_model = ActorModel(body_part, body_part_property, walker_body.obs_size)
        actor_model.eval()
        body_model[body_part] = actor_model

    checkpoint = torch.load(restore_path)

    for body_part, body_part_model in body_model.items():
        body_part_model.load_state_dict(checkpoint[body_part])

    channel = EngineConfigurationChannel()
    env = UnityEnvironment(walker_env_path, side_channels=[channel], worker_id=env_worker_id, no_graphics=False)
    #channel.set_configuration_parameters(time_scale=1.0)
    #channel.set_configuration_parameters(target_frame_rate=60)
    #channel.set_configuration_parameters(capture_frame_rate=True)
    env.reset()
    behavior_name = None
    for behavior_name in env.behavior_specs:
        print(behavior_name)
    action_spec = env.behavior_specs[behavior_name].action_spec

    # буферы для промежуточного хранения информации о статистике агентов
    agents_statistic = dict()

    step = 0
    time_4_step = 0.5
    while True:
        start_time = time.time()
        ds, ts = env.get_steps(behavior_name)
        if ds.agent_id.shape[0] > 0:
            n_agents = ds.agent_id.shape[0]
            continious_actions = torch.zeros(n_agents, action_spec.continuous_size)
            input_data = torch.from_numpy(ds.obs[0])
            for body_part, body_part_model in body_model.items():
                actions, _ = body_part_model.forward(input_data)
                idxs = walker_body.body[body_part].action_space_idxs
                continious_actions[:, idxs] = actions
            action_tuple = ActionTuple()
            action_tuple.add_continuous(continious_actions.numpy())

            env.set_actions(behavior_name, action_tuple)

        for agent_id in ds.agent_id:
            idx = ds.agent_id_to_index[agent_id]

            # суммируем и сохраняем статистику агента
            agent_statistic = agents_statistic.get(agent_id, dict())
            summary_reward = agent_statistic.get('summary_reward', 0.0)
            summary_reward += ds.reward[idx]
            agent_statistic['summary_reward'] = summary_reward
            if 'start_step' not in agent_statistic.keys():
                agent_statistic['start_step'] = step
            agents_statistic[agent_id] = agent_statistic

        # если есть агенты, для которых эпизод завершился
        if ts.agent_id.shape[0] > 0:
            for agent_id in ts.agent_id:
                idx = ts.agent_id_to_index[agent_id]
                reward = ts.reward[idx]

                total_reward = agents_statistic[agent_id]['summary_reward'] + reward
                total_steps = step - agents_statistic[agent_id]['start_step']
                print(f"Эпизод для агента {agent_id} закончен. Вознаграждение: {total_reward:.1f}, шагов: {total_steps}")

                # обнуляем статистику для агента, т.к. эпизод для него закончен
                agents_statistic[agent_id] = dict()

        step += 1
        env.step()
        # sleep_time = time_4_step - (time.time() - start_time)
        # if sleep_time > 0:
        #     time.sleep(sleep_time)



def run():
    # TODO: прокомментировать и переписать красиво

    args = sys.argv

    if '-walker_env_path' in args:
        idx = args.index('-walker_env_path')
        walker_env_path = args[idx + 1]
    else:
        walker_env_path = None

    if '-restore_path' in args:
        idx = args.index('-restore_path')
        restore_path = args[idx + 1]
    else:
        restore_path = None

    if '-env_worker_id' in args:
        idx = args.index('-env_worker_id')
        env_worker_id = int(args[idx + 1])
    else:
        env_worker_id = None

    run_model(walker_env_path=walker_env_path,
              restore_path=restore_path,
              env_worker_id=env_worker_id)


if __name__ == '__main__':
    run()
