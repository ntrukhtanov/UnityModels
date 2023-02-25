from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import torch

from body_parts import WalkerBody
from actor import ActorModel

import sys
import time
from datetime import timedelta


def run_model(walker_env_path, restore_path, env_worker_id, break_body_parts=None, break_body_parts_period=None):
    walker_body = WalkerBody()

    # если нужно протестировать агентов со сломанными частями тела, то соберем соответствующие индексы
    # в пространствах наблюдений и действий для последующего использования
    obs_brake_idxs = list()
    actions_breake_idxs = list()
    if break_body_parts is not None:
        broken_body_parts = break_body_parts.split(",")
        for body_part in broken_body_parts:
            body_part_prop = walker_body.body[body_part]
            obs_brake_idxs.extend(
                [idx for idx in body_part_prop.obs_space_idxs if idx not in walker_body.common_obs_space_idxs])
            actions_breake_idxs.extend(body_part_prop.action_space_idxs)

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
    #channel.set_configuration_parameters(target_frame_rate=24)
    #channel.set_configuration_parameters(capture_frame_rate=True)
    env.reset()
    behavior_name = None
    for behavior_name in env.behavior_specs:
        print(behavior_name)
    action_spec = env.behavior_specs[behavior_name].action_spec

    # буферы для промежуточного хранения информации о статистике агентов
    agents_statistic = dict()
    step = 0
    start_time = time.time()
    time_2_break_body_parts = break_body_parts_period is None
    while True:
        ds, ts = env.get_steps(behavior_name)
        if ds.agent_id.shape[0] > 0:
            n_agents = ds.agent_id.shape[0]
            continious_actions = torch.zeros(n_agents, action_spec.continuous_size)
            input_data = torch.from_numpy(ds.obs[0])

            # если есть сломанные части тела, то обнулим информацию о наблюдениях для этих частей тела
            if len(obs_brake_idxs) > 0 and time_2_break_body_parts:
                input_data[:, obs_brake_idxs] = 0.0

            for body_part, body_part_model in body_model.items():
                actions, _ = body_part_model.forward(input_data)
                idxs = walker_body.body[body_part].action_space_idxs
                continious_actions[:, idxs] = actions

            # если есть сломанные части тела, то обнулим информацию о действиях для этих частей тела
            if len(actions_breake_idxs) > 0 and time_2_break_body_parts:
                continious_actions[:, actions_breake_idxs] = 0.0

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

        if break_body_parts_period is not None and \
                (time.time() - start_time) > timedelta(seconds=break_body_parts_period).seconds:
            time_2_break_body_parts = ~time_2_break_body_parts
            start_time = time.time()


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

    if '-break_body_parts' in args:
        idx = args.index('-break_body_parts')
        break_body_parts = args[idx + 1]
    else:
        break_body_parts = None

    if '-break_body_parts_period' in args:
        idx = args.index('-break_body_parts_period')
        break_body_parts_period = int(args[idx + 1])
    else:
        break_body_parts_period = None

    run_model(walker_env_path=walker_env_path,
              restore_path=restore_path,
              env_worker_id=env_worker_id,
              break_body_parts=break_body_parts,
              break_body_parts_period=break_body_parts_period)


if __name__ == '__main__':
    run()
