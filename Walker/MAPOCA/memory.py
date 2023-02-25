import random

import torch


class ExperienceBuffer:
    def __init__(self, walker_body, agent_ids, buffer_size):
        """
        Буфер для хранения информации о траекториях агентов
        :param walker_body: информация о декомпозиции агента
        :param agent_ids: идентификаторы агентов в среде Walker
        :param buffer_size: размер буфера
        """
        self.walker_body = walker_body
        self.agent_ids = agent_ids
        self.buffer_size = buffer_size

        self.buffer = dict()

        self.reset()

    def reset(self):
        """
        Метод сброса состояния буфера в начальное состояние
        :return:
        """
        self.buffer = dict()
        for agent_id in self.agent_ids:
            self.buffer[agent_id] = dict()
            self.buffer[agent_id]["obs"] = list()  # наблюдения агента
            self.buffer[agent_id]["actions"] = list()  # действия агента
            self.buffer[agent_id]["log_probs"] = list()  # логарифмы вероятностей действий агента
            self.buffer[agent_id]["rewards"] = list()  # вознаграждения агента
            self.buffer[agent_id]['next_obs'] = list()  # наблюдения агента после выполнения действий
            self.buffer[agent_id]['dones'] = list()  # признаки завершения эпизода для агента

            self.buffer[agent_id]["common_values"] = list()  # значения выходов модели критика на основе наблюдений
            for body_part in self.walker_body.body.keys():
                self.buffer[agent_id][body_part] = dict()
                # значения выходов модели критика на основе наблюдений текущей части тела
                # и наблюдений + действий остальных частей тела агента
                self.buffer[agent_id][body_part]["body_part_values"] = list()

    def add_experience(self, agent_id, obs, actions, reward, log_probs):
        """
        Метод добавления записи в буфер
        :param agent_id: id агента в среде Walker
        :param obs: наблюдения агента
        :param actions: действия агента
        :param reward: вознаграждение агента
        :param log_probs: логарифмы вероятностей действий агента
        :return:
        """
        if len(self.buffer[agent_id]["obs"]) > len(self.buffer[agent_id]["next_obs"]):
            self.buffer[agent_id]["next_obs"].append(obs)
            self.buffer[agent_id]["rewards"].append(reward)
            self.buffer[agent_id]["dones"].append(False)
        self.buffer[agent_id]["obs"].append(obs)
        self.buffer[agent_id]["actions"].append(actions)
        self.buffer[agent_id]["log_probs"].append(log_probs)

    def set_terminate_state(self, agent_id, obs, reward):
        """
        Метод добавления информации в момент завершения эпизода для агента
        :param agent_id: id агента в среде Walker
        :param obs: наблюдения агента
        :param reward: вознаграждение агента
        :return:
        """
        if len(self.buffer[agent_id]["obs"]) > 0:
            self.buffer[agent_id]["next_obs"].append(obs)
            self.buffer[agent_id]["rewards"].append(reward)
            self.buffer[agent_id]["dones"].append(True)

    def add_common_values(self, values):
        """
        Метод добавления данных о выходе модели критика на основе наблюдений всех частей тела агента
        :param values: значения на выходе модели критика
        :return:
        """
        for i, agent_id in enumerate(self.agent_ids):
            self.buffer[agent_id]["common_values"].append(values[i])

    def add_body_part_values(self, body_part, values):
        """
        Метод добавления данных о выходе модели критика на основе наблюдений текущей части тела и
        конкатенации наблюдений и действий остальных частей тела
        :param body_part: текущая часть тела
        :param values: значения на выходе модели критика
        :return:
        """
        for i, agent_id in enumerate(self.agent_ids):
            self.buffer[agent_id][body_part]["body_part_values"].append(values[i])

    def add_returns(self, agent_id, returns):
        """
        Функция сохранения в буфер рассчитанных значений целевой функции полезности
        :param agent_id: id агента в среде Walker
        :param returns: значения целевой функции полезности
        :return:
        """
        self.buffer[agent_id]["returns"] = returns

    def add_advantages(self, agent_id, body_part, advantages):
        """
        Функция сохранения в буфер значений функции преимущества
        :param agent_id: id агента в среде Walker
        :param body_part: часть тела агента для которой рассчитана функция преимущества
        :param advantages: значения функции преимущества
        :return:
        """
        self.buffer[agent_id][body_part]["advantages"] = advantages

    def sample_buffer(self, shuffle):
        """
        Извлечение данных буфера в удобной для обучения форме с перемешиванием данных
        :param shuffle: признак необходимости перемешать данные
        :return: буфер с подготовленными данными
        """
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
        """
        Извлечение из буфера батча нужного размера
        :param buffer: Буфер из которого нужно извлечь батч
        :param batch_step: Индекс в буфере с которого начать считыванеи батча
        :param batch_size: Размер батча
        :param device: Устройство в которое следует отправить данные
        :param device_keys: Ключи значний в буфере, которые следует отправить на устройстов
        (не все нужно отправлять в GPU)
        :return: Батч ч данными
        """
        batch = dict()
        for body_part in self.walker_body.body.keys():
            batch[body_part] = dict()
            for k in buffer[body_part].keys():
                batch[body_part][k] = buffer[body_part][k][batch_step:batch_step+batch_size]
                if k in device_keys:
                    batch[body_part][k] = batch[body_part][k].to(device)
        return batch

    def get_last_record_batch(self, device):
        """
        Метод извлечения последней записи в буфере
        :param device: Устройстов на которое следует отправить извлеченные данные
        :return: Буфер с последней записью
        """
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
        """
        Метод проверки, что буфер наполнился
        :return: True/False
        """
        for agent_id in self.agent_ids:
            if len(self.buffer[agent_id]["actions"]) < self.buffer_size + 1:
                return False
        return True
