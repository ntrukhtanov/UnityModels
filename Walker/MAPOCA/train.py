from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

import numpy as np
import torch
import random
from torch.utils.tensorboard import SummaryWriter
import sys
from tqdm import tqdm
import statistics
import os
import traceback
import math
import time

from body_parts import WalkerBody
from body_parts import BodyPartProperties

from actor import ActorModel
from critic import CriticModel
from loss_functions import calc_value_loss
from loss_functions import calc_returns
from loss_functions import calc_policy_loss
from memory import ExperienceBuffer
from cloud_saver import CloudSaver
from eval import evaluate


RANDOM_SEED = 42

GAMMA = 0.99
LAM = 0.95
EPSILON = 0.2
BETA = 0.01


def train(walker_env_path, summary_dir, total_steps, buffer_size, batch_size, iter_count,
          save_path=None, save_freq=None, restore_path=None, cloud_path=None, env_worker_id=None,
          cloud_restore_path=None, eval_freq=None):
    """
    Функция обучения модели.
    :param walker_env_path: Путь к сборке среды Walker
    :param summary_dir: Путь к папке tensorboard
    :param total_steps: Количество шагов обучения
    :param buffer_size: Размер буфера для хранения траекторий агентов и расчитываемых перемнных
    :param batch_size: Размер батча для обучения
    :param iter_count: Количество прогонов обучения для каждого батча
    :param save_path: Путь к папке, куда сохраняются модели, для последующего восстановления
    :param save_freq: Частота сохранения модели (каждые save_freq шагов)
    :param restore_path: Путь к файлу модели для восстановления
    :param cloud_path: Путь в облачном хранилище для копирования модели
    :param env_worker_id: id пространства unity Walker
    :param cloud_restore_path: Путь в облачном хранилище для загрузки модели
    :param eval_freq: Частота тестового прогона модели (каждые eval_freq шагов)
    :return:
    """
    assert walker_env_path is not None, f"Не указан обязательный параметр walker_env_path"
    assert summary_dir is not None, f"Не указан обязательный параметр summary_dir"
    assert total_steps is not None, f"Не указан обязательный параметр total_steps"
    assert buffer_size is not None, f"Не указан обязательный параметр buffer_size"
    assert batch_size is not None, f"Не указан обязательный параметр batch_size"
    assert iter_count is not None, f"Не указан обязательный параметр iter_count"

    # размер буфера должен делиться на размер батча без остатка
    buffer_size = math.ceil(buffer_size / batch_size) * batch_size

    # Установим состояния генераторов псевдослучайных чисел для воспризводимости и сравнимости результатов
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    # определяем текущее устройство
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # инициализируем экземпляр класса записи данных для tensorboard
    summary_writer = SummaryWriter(summary_dir)

    # если путь в облачном хранилище определн, то инициализируем класс копирования данных в облако
    cloud_saver = None
    if cloud_path is not None:
        cloud_saver = CloudSaver(cloud_path)

    # инициализируем вспопогательный класс WalkerBody
    # класс хранит в себе данные о декомпоизии агента на условную команду,
    # состоящую и частей тела агента
    walker_body = WalkerBody()

    # инициализация моделей актора и критика
    # в зависимости от значения параметра restore_path модель восстанавивается из файла или только инициализируется
    params = list()
    body_model = dict()

    # проходим по всем частям тела агента и инициализиурем модель актора для каждой части тела агента
    for body_part, body_part_property in walker_body.body.items():
        actor_model = ActorModel(body_part, body_part_property, walker_body.obs_size).to(device)
        params.extend(list(actor_model.parameters()))
        body_model[body_part] = actor_model

    # создаем модель критика
    critic_model = CriticModel(walker_body).to(device)
    params.extend(list(critic_model.parameters()))

    # инициализируем оптимизатор
    optimizer = torch.optim.Adam(params, lr=0.0003)

    # инициализируем шаг обучения
    step = 0

    # если указан путь к файлу восстановления параметров модели, то обновляем параметры
    if restore_path is not None or cloud_restore_path is not None:
        if cloud_restore_path is not None:
            restore_path = os.path.join(save_path, os.path.basename(cloud_restore_path))
            cloud_saver.download_model(cloud_restore_path, restore_path)
        checkpoint = torch.load(restore_path, map_location=device)

        for body_part, body_part_model in body_model.items():
            body_part_model.load_state_dict(checkpoint[body_part])

        critic_model.load_state_dict(checkpoint['critic_model'])

        optimizer.load_state_dict(checkpoint['optimizer'])

        step = checkpoint['step'] + 1

    # переводим модели в состояение обучения
    for body_part_model in body_model.values():
        body_part_model.train()

    critic_model.train()

    # инициализируем среду Walker без графического представления для скорости
    if env_worker_id is None:
        env_worker_id = 1
    env = UnityEnvironment(walker_env_path, worker_id=env_worker_id, no_graphics=True)

    # инициализируем среду Walker без графического представления для тестирования модели
    eval_env = None
    if eval_freq is not None:
        eval_env = UnityEnvironment(walker_env_path, worker_id=env_worker_id+1, no_graphics=True)

    # сбрасываем среду в начальное состояние
    env.reset()

    # определяем значение ключа behavior_name, по нему будем получать данные о среде
    behavior_name = None
    for behavior_name in env.behavior_specs:
        print(behavior_name)

    # получим описание пространства действий агента
    # в среде Walker мы имеем 39 непрерывных действий и 0 дискретных действий
    action_spec = env.behavior_specs[behavior_name].action_spec

    # получим описание пространства наблюдений среды со стороны агента
    # в среде Walker мы имеем тензор наблюдений агента, состоящий из 243 чисел
    observation_specs = env.behavior_specs[behavior_name].observation_specs

    # объект памяти для хранения данных о траекториях агентов и рассчитываемых переменных
    memory = None

    # буферы для промежуточного хранения информации о статистике агентов
    agents_statistic = dict()
    agents_total_rewards = list()
    agents_lifetimes = list()

    # инициализируем объект tqdm для отображения прогресса обучения по шагам
    # немного не стандартная инициализация, нужна, чтобы начинать не с первого шага
    # в случае восстановления модели из бэкапа
    pbar = tqdm(range(total_steps))
    pbar.update(n=step)
    actor_start_time = time.time()
    while step < total_steps:
        # получаем данные из среды Walker:
        # ds - содержит данные об агентах, которые требуют указания действий для следующего шага
        # ts - содержит данные об агентах, которые достигли конца эпизода
        ds, ts = env.get_steps(behavior_name)

        # если есть агенты, требующие решения о следующих действиях
        if ds.agent_id.shape[0] > 0:
            # если объект памяти еще не создан, то создаем его
            if memory is None:
                memory = ExperienceBuffer(walker_body, ds.agent_id, buffer_size)

            # определяем сколько агентов требуют действий
            # среда обрабатывает сразу 10 агентов, но часть из них может завершить эпизод
            n_agents = ds.agent_id.shape[0]

            # создаем тензор для непрерывных действий, заполненный нулями
            continious_actions = torch.zeros(n_agents, action_spec.continuous_size)

            # преобразуем матрицу наблюдений в тензор
            input_data = torch.from_numpy(ds.obs[0])

            # инициализируем тензор для сохранения логарифмов вероятностей действий
            # эти данные потребуются для вычисления функции потерь для моделей актора
            log_probs = torch.zeros(n_agents, action_spec.continuous_size)

            # проходим по каждой части тела и вычисляем для них действия
            # для вычислений используем текущие модели акторов
            for body_part, body_part_model in body_model.items():
                # градиенты вычислять не нужно, будем их вычислять в момент обучения
                with torch.no_grad():
                    # отправляем данные на устройство
                    input_data = input_data.to(device)
                    # вычисляем действия и логарифмы вероятностей действий
                    # на вход модели подаем все пространство наблюдений
                    # это логично, потому что для решения о том, какое действие предпринять, например, ноге,
                    # нужно знать состояние других частей тела
                    actions, log_prob = body_part_model.forward(input_data)

                # сохраняем полученные действия в общий массив действий агента на позиции,
                # которые соответствуют данной части тела
                idxs = walker_body.body[body_part].action_space_idxs
                continious_actions[:, idxs] = actions.cpu().detach()

                # аналогично сохраняем логарифмы вероятностей действий
                log_probs[:, idxs] = log_prob.cpu().detach()

            # формируем объект ActionTuple, который требуется среде Walker для обработки действий
            action_tuple = ActionTuple()
            action_tuple.add_continuous(continious_actions.detach().cpu().numpy())

            # отправляем действия в среду Walker
            env.set_actions(behavior_name, action_tuple)

            # для каждого агента сохраняем данные в буфер памяти
            for agent_id in ds.agent_id:
                idx = ds.agent_id_to_index[agent_id]
                # параметры
                # agent_id: идентификатор агента
                # obs: тензор наблюдений для агента
                # actions: тензор действий для агента
                # reward: вознаграждение агента
                # done: признак завершения эпизода для агента
                # log_probs: логарифмы вероятностей действий агента
                memory.add_experience(agent_id=agent_id, obs=torch.from_numpy(ds.obs[0][idx]),
                                      actions=continious_actions[idx],
                                      reward=ds.reward[idx], log_probs=log_probs[idx])

                # суммируем и сохраняем статистику агента
                agent_statistic = agents_statistic.get(agent_id, dict())
                summary_reward = agent_statistic.get('summary_reward', 0.0)
                summary_reward += ds.reward[idx]
                agent_statistic['summary_reward'] = summary_reward
                if 'start_step' not in agent_statistic.keys():
                    agent_statistic['start_step'] = step
                agents_statistic[agent_id] = agent_statistic

            # Для новой порции данных вычисляем значения модели критика.
            # Эти значения понадобятся для расчета функции потерь.

            # Сформируем батч из последних значений траекторий агентов
            batch = memory.get_last_record_batch(device)
            with torch.no_grad():
                # вычислим значения модели критика на основании наблюдений всех частей тела агента, без учета их действий
                common_values = critic_model.critic_common(batch)
                memory.add_common_values(common_values.cpu())

                # вычислим значения модели критика на основании наблюдений всех частей тела,
                # с учетом действий частей тела отличных от текущего
                # будем выполнять вычисления для каждой части тела агента по отдельности таким образом,
                # так как если бы это были отдельные агенты, принадлежащие одной команде
                for body_part in walker_body.body.keys():
                    values_body_part = critic_model.critic_body_part(batch, body_part)
                    memory.add_body_part_values(body_part, values_body_part.cpu())

        # если есть агенты, для которых эпизод завершился
        if ts.agent_id.shape[0] > 0:
            for agent_id in ts.agent_id:
                idx = ts.agent_id_to_index[agent_id]
                obs = torch.from_numpy(ts.obs[0][idx])
                reward = ts.reward[idx]

                # сохраняем наблюдение и вознаграждение
                memory.set_terminate_state(agent_id, obs, reward)

                # добавляем суммарное вознаграждение в список, для последующего усреднения и отправки в tensorboard
                agents_total_rewards.append(agents_statistic[agent_id]['summary_reward'] + reward)

                # добавляем количество шагов жизни агента, для последующего усреднения и отправки в tensorboard
                agents_lifetimes.append(step - agents_statistic[agent_id]['start_step'])

                # обнуляем статистику для агента, т.к. эпизод для него закончен
                agents_statistic[agent_id] = dict()

        # просим среду Walker отработать наши действия
        env.step()

        # если буфер траекторий агентов наполнился, то начинаем обучать модель
        if memory.buffer_is_full():
            print(f"Actor run time,s: {(time.time() - actor_start_time):.1f}")
            train_start_time = time.time()
            # сначала вычислим необходимые переменные на текущей модели и добавим их в память
            # эти значения потребуются для вычисления функций потерь во время обучения
            for agent_id in memory.agent_ids:
                common_values = torch.Tensor(memory.buffer[agent_id]["common_values"])

                # вычислим дисконтированные вознаграждения с учетом ранее вычисленных значений модели критика
                rewards = torch.Tensor(memory.buffer[agent_id]["rewards"])[:buffer_size]
                dones = torch.Tensor(memory.buffer[agent_id]["dones"])[:buffer_size]
                returns = calc_returns(rewards=rewards,
                                       dones=dones,
                                       values=common_values[:buffer_size],
                                       gamma=GAMMA,
                                       lam=LAM,
                                       value_next=common_values[buffer_size])
                returns = returns.unsqueeze(1)

                # добавим их в буфер для последующего расчета функции потерь
                memory.add_returns(agent_id, returns)

                # вычислим значения функции преимущества отдельно для каждой части тела агента и добавим в буфер
                for body_part in walker_body.body.keys():
                    body_part_values = torch.Tensor(memory.buffer[agent_id][body_part]["body_part_values"])[:buffer_size]
                    body_part_values = body_part_values.unsqueeze(1)
                    advantages = returns - body_part_values
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)
                    memory.add_advantages(agent_id, body_part, advantages)

            # обучение модели
            # инициализируем списки для промежуточного хранения значений функций потерь
            # для последующего усреднения и отправки в tensorboard
            losses = list()
            policy_losses = list()
            value_losses = list()
            value_body_losses = list()
            entropies = list()
            # обучение будем проводить iter_count итераций для накопленной выборки данных
            for iter in range(iter_count):
                # извлекаем данные из памяти, перемешивая их по траекториям всех агентов, но не по частям тела агента
                buffer = memory.sample_buffer(shuffle=True)
                for batch_step in range(0, buffer['size'], batch_size):
                    # вырежем из буфера текущий батч
                    batch = memory.sample_batch(buffer, batch_step, batch_size, device,
                                                ['full_obs', 'obs', 'actions'])

                    # перемешаем список частей тела агента
                    body_part_names = list(walker_body.body.keys())
                    random.shuffle(body_part_names)
                    for body_part in body_part_names:
                        # вычислим логарифмы вероятностей действий и энтропию для каждой части тела агента
                        # с помощью модели актора для соответствующей части тела
                        full_obs = batch[body_part]["full_obs"]
                        actions = batch[body_part]["actions"]
                        log_probs, entropy = body_model[body_part].evaluate_actions(full_obs, actions)

                        # вычислим значения текущей модели критика на основании наблюдений всех частей тела, без учета действий
                        # в данном случае вычисляем эти значения для каждой части тела,
                        # потому что модель меняется в результате обучения
                        common_values = critic_model.critic_common(batch)

                        # вычислим значения для текущей модели критика на основании наблюдений всех частей тела,
                        # с учетом действий частей тела отличных от текущего
                        body_part_values = critic_model.critic_body_part(batch, body_part)

                        # извлечем ранее рассчитанные значения
                        old_common_values = batch[body_part]["common_values"]
                        old_body_part_values = batch[body_part]["body_part_values"]
                        returns = batch[body_part]["returns"]
                        old_log_probs = batch[body_part]["log_probs"]
                        advantages = batch[body_part]["advantages"]

                        # рассчитаем функции потерь для модели критика, используя значения,
                        # полученные с помощью старой модели и с помощью текущей модели
                        value_loss = calc_value_loss(common_values.cpu(), old_common_values, returns, EPSILON)
                        value_body_loss = calc_value_loss(body_part_values.cpu(), old_body_part_values, returns, EPSILON)

                        # рассчитаем функцию потерь для актора текущей части тела на основании рассчитанных
                        # с помощью старой модели значений переменных преимущества и логарифмов вероятностей действий,
                        # и рассчитанных с помощью текущей модели значений логарифмов вероятностей действий
                        policy_loss = calc_policy_loss(advantages, log_probs.cpu(), old_log_probs, EPSILON)

                        # суммируем все функции потерь в одну
                        loss = (
                                policy_loss
                                + 0.5 * (value_loss + 0.5 * value_body_loss)
                                - BETA * torch.mean(entropy.cpu())
                        )

                        # сохраняем значения функции потерь для статистики
                        losses.append(loss.item())
                        policy_losses.append(policy_loss.item())
                        value_losses.append(value_loss.item())
                        value_body_losses.append(value_body_loss.item())
                        entropies.append(torch.mean(entropy).item())

                        # обнуляем и вычисляем градиенты
                        optimizer.zero_grad()
                        loss.backward()

                        # выполняем шаг обучения
                        optimizer.step()

            # отправляем усредненные значения функций потерь в tensorboard
            summary_writer.add_scalar("loss", statistics.mean(losses), step)
            summary_writer.add_scalar("policy_loss", statistics.mean(policy_losses), step)
            summary_writer.add_scalar("value_loss", statistics.mean(value_losses), step)
            summary_writer.add_scalar("value_body_loss", statistics.mean(value_body_losses), step)
            summary_writer.add_scalar("entropy", statistics.mean(entropies), step)

            # усредняем и отправляем усредненные суммарные вознаграждения в tensorboard
            summary_writer.add_scalar("Total reward", statistics.mean(agents_total_rewards), step)
            agents_total_rewards.clear()

            # усредняем и отправляем усредненные значения количества шагов жизни агента в tensorboard
            summary_writer.add_scalar("Agents lifetimes", statistics.mean(agents_lifetimes), step)
            agents_lifetimes.clear()

            # сбрасываем буфер траекторий, т.к. у нас on-policy алгоритм
            # и нам нужно накопить данные уже на основании только что обученной модели
            memory = None

            # сбрасываем буферы для промежуточного хранения информации о статистике агентов
            agents_statistic = dict()
            agents_total_rewards = list()
            agents_lifetimes = list()

            # сбрасываем среду в начальное состояние
            env.reset()

            print(f"Train time,s: {(time.time() - train_start_time):.1f}")
            actor_start_time = time.time()

        # если достигли шага, на котором нужно протестировать модель, то тестируем и пишем результаты в tensorboard
        if eval_freq is not None and step > 0 and step % eval_freq == 0:
            eval_start_time = time.time()

            eval_total_reward, eval_agents_lifetime = evaluate(walker_body, eval_env, body_model, 25, device)
            summary_writer.add_scalar("Eval total reward", eval_total_reward, step)
            summary_writer.add_scalar("Eval agents lifetimes", eval_agents_lifetime, step)

            # возвращаем модель в состояение обучения
            for body_part_model in body_model.values():
                body_part_model.train()

            print(f"Eval time,s: {(time.time() - eval_start_time):.1f}")

        # если достигли шага, на котором нужно сохранять модель, то сохраняем
        if save_freq is not None and save_path is not None:
            if step > 0 and step % save_freq == 0:
                save_dict = dict()
                save_dict['step'] = step
                for body_part, body_part_model in body_model.items():
                    save_dict[body_part] = body_part_model.state_dict()
                save_dict['critic_model'] = critic_model.state_dict()
                save_dict['optimizer'] = optimizer.state_dict()
                save_file_name = os.path.join(save_path, f"model_{step}.pt")
                torch.save(save_dict, save_file_name)

                summary_writer.flush()

                if cloud_saver is not None:
                    summary_writer.close()
                    try:
                        cloud_saver.save(checkpoint_file_name=save_file_name, tensorboard_dir=summary_dir)
                    except Exception as ex:
                        print(f"Ошибка копирования состояния в облако")
                        for arg in ex.args:
                            print(str(arg))
                        print(traceback.format_exc())

        # увеличиваем шаг на 1 и обновляем прогрессбар
        step += 1
        pbar.update(1)


def run():
    # TODO: прокомментировать и переписать красиво
    # -buffer_size 32
    # -total_steps 30000000
    # -walker_env_path ./Unity/Walker
    # -summary_dir /home/tnv/tensorboard
    # -save_path /home/tnv/tempModel
    # -save_freq 10000
    # -restore_path /home/tnv/tempModel/model_10.pt
    # -cloud_path /Kaggle/Walker

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

    if '-batch_size' in args:
        idx = args.index('-batch_size')
        batch_size = int(args[idx + 1])
    else:
        batch_size = None

    if '-iter_count' in args:
        idx = args.index('-iter_count')
        iter_count = int(args[idx + 1])
    else:
        iter_count = None

    if '-walker_env_path' in args:
        idx = args.index('-walker_env_path')
        walker_env_path = args[idx + 1]
    else:
        walker_env_path = None

    if '-save_path' in args:
        idx = args.index('-save_path')
        save_path = args[idx + 1]
    else:
        save_path = None

    if '-save_freq' in args:
        idx = args.index('-save_freq')
        save_freq = int(args[idx + 1])
    else:
        save_freq = None

    if '-restore_path' in args:
        idx = args.index('-restore_path')
        restore_path = args[idx + 1]
    else:
        restore_path = None

    if '-cloud_path' in args:
        idx = args.index('-cloud_path')
        cloud_path = args[idx + 1]
    else:
        cloud_path = None

    if '-env_worker_id' in args:
        idx = args.index('-env_worker_id')
        env_worker_id = int(args[idx + 1])
    else:
        env_worker_id = None

    if '-cloud_restore_path' in args:
        idx = args.index('-cloud_restore_path')
        cloud_restore_path = args[idx + 1]
    else:
        cloud_restore_path = None

    if '-eval_freq' in args:
        idx = args.index('-eval_freq')
        eval_freq = int(args[idx + 1])
    else:
        eval_freq = None

    train(walker_env_path=walker_env_path,
          summary_dir=summary_dir,
          total_steps=total_steps,
          buffer_size=buffer_size,
          batch_size=batch_size,
          iter_count=iter_count,
          save_path=save_path,
          save_freq=save_freq,
          restore_path=restore_path,
          cloud_path=cloud_path,
          env_worker_id=env_worker_id,
          cloud_restore_path=cloud_restore_path,
          eval_freq=eval_freq)


if __name__ == '__main__':
    run()

