from mlagents_envs.base_env import ActionTuple
import torch
from statistics import mean


def evaluate(walker_body, env, model, episodes, device, broken_body_parts):
    """
    Метод тестирования модели без отрисовки среды
    :param walker_body: данные о декомпозиции агента
    :param env: объект среды Walker для тестирования
    :param model: модель актора
    :param episodes: количество эпизодов для тестирования
    :param device: устройство на котором выполнять актор
    :param broken_body_parts: сломанные части тела агента
    :return:
    """
    # инициализируем среду Walker
    env.reset()

    # если нужно протестировать агентов со сломанными частями тела, то соберем соответствующие индексы
    # в пространствах наблюдений и действий для последующего использования
    obs_brake_idxs = list()
    actions_breake_idxs = list()
    if broken_body_parts is not None:
        for body_part in broken_body_parts:
            body_part_prop = walker_body.body[body_part]
            obs_brake_idxs.extend(
                [idx for idx in body_part_prop.obs_space_idxs if idx not in walker_body.common_obs_space_idxs])
            actions_breake_idxs.extend(body_part_prop.action_space_idxs)

    # переводим модель в состояение тестирования
    for body_part_model in model.values():
        body_part_model.eval()

    # определяем значение ключа behavior_name, по нему будем получать данные о среде
    for behavior_name in env.behavior_specs:
        break

    # получим описание пространства действий агента
    # в среде Walker мы имеем 39 непрерывных действий и 0 дискретных действий
    action_spec = env.behavior_specs[behavior_name].action_spec

    # буферы для промежуточного хранения информации о статистике агентов
    agents_statistic = dict()
    agents_total_rewards = list()
    agents_lifetimes = list()

    step = 0
    while len(agents_total_rewards) < episodes:
        # получаем данные из среды Walker:
        # ds - содержит данные об агентах, которые требуют указания действий для следующего шага
        # ts - содержит данные об агентах, которые достигли конца эпизода
        ds, ts = env.get_steps(behavior_name)

        # если есть агенты, требующие решения о следующих действиях
        if ds.agent_id.shape[0] > 0:
            # определяем сколько агентов требуют действий
            # среда обрабатывает сразу 10 агентов, но часть из них может завершить эпизод
            n_agents = ds.agent_id.shape[0]

            # преобразуем матрицу наблюдений в тензор
            input_data = torch.from_numpy(ds.obs[0])

            # создаем тензор для непрерывных действий, заполненный нулями
            continious_actions = torch.zeros(n_agents, action_spec.continuous_size)

            # если есть сломанные части тела, то обнулим информацию о наблюдениях для этих частей тела
            if len(obs_brake_idxs) > 0:
                input_data[:, obs_brake_idxs] = 0.0

            # проходим по каждой части тела и вычисляем для них действия
            # для вычислений используем текущие модели акторов
            for body_part, body_part_model in model.items():
                # отправляем данные на устройство
                input_data = input_data.to(device)
                # вычисляем действия и логарифмы вероятностей действий
                # на вход модели подаем все пространство наблюдений
                # это логично, потому что для решения о том, какое действие предпринять, например, ноге,
                # нужно знать состояние других частей тела
                actions, _ = body_part_model.forward(input_data)

                # сохраняем полученные действия в общий массив действий агента на позиции,
                # которые соответствуют данной части тела
                idxs = walker_body.body[body_part].action_space_idxs
                continious_actions[:, idxs] = actions.cpu()

            # если есть сломанные части тела, то обнулим информацию о действиях для этих частей тела
            if len(actions_breake_idxs) > 0:
                continious_actions[:, actions_breake_idxs] = 0.0

            # формируем объект ActionTuple, который требуется среде Walker для обработки действий
            action_tuple = ActionTuple()
            action_tuple.add_continuous(continious_actions.cpu().numpy())

            # отправляем действия в среду Walker
            env.set_actions(behavior_name, action_tuple)

            # для каждого агента сохраняем статистику
            for agent_id in ds.agent_id:
                idx = ds.agent_id_to_index[agent_id]

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

                # добавляем суммарное вознаграждение в список, для последующего усреднения и отправки в tensorboard
                agents_total_rewards.append(agents_statistic[agent_id]['summary_reward'] + reward)

                # добавляем количество шагов жизни агента, для последующего усреднения и отправки в tensorboard
                agents_lifetimes.append(step - agents_statistic[agent_id]['start_step'])

                # обнуляем статистику для агента, т.к. эпизод для него закончен
                agents_statistic[agent_id] = dict()

        # просим среду Walker отработать наши действия
        env.step()

        step += 1

    eval_total_reward = mean(agents_total_rewards)
    eval_agents_lifetime = mean(agents_lifetimes)

    return eval_total_reward, eval_agents_lifetime
