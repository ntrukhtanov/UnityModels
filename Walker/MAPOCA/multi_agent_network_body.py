import torch
from torch import nn

from linear_encoder import LinearEncoder
from entity_embedding import EntityEmbedding
from attention import ResidualSelfAttention

EMBEDDING_SIZE = 64  # в оригинале 256


class MultiAgentNetworkBody(nn.Module):
    def __init__(self, walker_body, hidden_size):
        """
        Модель критика для нескольких агентов
        :param walker_body: описание структуры тела агента Walker
        :param hidden_size: размер скрытого слоя для выходного энкодера
        """
        super().__init__()

        self.walker_body = walker_body

        # Пройдем по всем частям тела агента и проинициализируем модель, которая обучает соответствующий embedding.
        # В общем случае то позволит подать на вход критика разнородных агентов.
        # В нашем случае разнородные агенты - это части тела агента.
        # Нам потребуется по два эмбединга на каждую часть тела,
        # т.к. алгоритм подразумевает кодирование наблюдений агентов без действий и с действиями.
        self.obs_encoders = dict()
        self.obs_action_encoders = dict()
        for body_part, body_part_prop in walker_body.body.items():
            # создаем модель эмбединга для текущей части тела с входом только для наблюдений
            self.obs_encoders[body_part] = EntityEmbedding(body_part_prop.input_dim, EMBEDDING_SIZE)
            self.add_module(f"obs_encoder_{body_part}", self.obs_encoders[body_part])
            # создаем модель эмбединга для текущей части тела с входом для наблюдений + действий
            self.obs_action_encoders[body_part] = EntityEmbedding(
                body_part_prop.input_dim + body_part_prop.output_dim, EMBEDDING_SIZE)
            self.add_module(f"obs_action_encoder_{body_part}", self.obs_action_encoders[body_part])

        # инициализируем модель, реализующую механизм внимания
        self.self_attn = ResidualSelfAttention(EMBEDDING_SIZE)

        # инициализируем модель из полносвязных слоев, которая будет являться выходом данной модели
        self.linear_encoder = LinearEncoder(EMBEDDING_SIZE, 1, hidden_size)

    def forward(self, batch, body_part):
        """
        Функция прямого прохождения модели критика для нескольких агентов
        :param batch: батч с данными
        :param body_part: часть тела агента для которой выполняются вычисления.
        Может быть None для вычисления функции потерь без учета действий других частей тела агента.
        :return: Возвращает значения модели критика для нескольких агентов
        """

        # список, в котором собираем результаты работы энкодеров, для дальнейшей обработки с помощью механизма внимания
        self_attn_inputs = list()

        if body_part is None:
            # вычисляем результаты работы энкодеров на основании наблюдений всех частей тела, без учета действий
            all_encoded_obs = list()
            for body_part in self.walker_body.body.keys():
                # Для каждой части тела свой энкодер наблюдений, т.к. наблюдения имеют разную размерность
                # и части тела выполняют разные функции.
                all_encoded_obs.append(self.obs_encoders[body_part](batch[body_part]["obs"]))

            # собираем все в один тензор и добавляем в список для механизма внимания
            all_encoded_obs = torch.stack(all_encoded_obs, dim=1)
            self_attn_inputs.append(all_encoded_obs)
        else:
            # Вычислим результаты работы энкодеров на основании наблюдения текущей части тела (параметр body_part)
            # без учета его действий,
            # и на основании наблюдений всех остальных частей тела, с учетом их действий

            # вычисляем энкодер для текущй части тела и добавляем в список на вход механизма внимания
            encoded_obs = self.obs_encoders[body_part](batch[body_part]["obs"])
            encoded_obs = torch.stack([encoded_obs], dim=1)
            self_attn_inputs.append(encoded_obs)

            # проходим по всем частям тела кроме текущей и вычисляем энкодеры с учетом их действий
            encoded_obs_with_actions = list()
            for other_body_part in self.walker_body.body.keys():
                if other_body_part != body_part:
                    obs = batch[other_body_part]["obs"]
                    actions = batch[other_body_part]["actions"]
                    # объединяем наблюдения и действия в один тензор
                    obs_with_actions = torch.cat([obs, actions], dim=1)

                    # для каждой части тела свой отдельный энкодер,
                    # т.к. пространства наблюдений и действий разнородны и функции частей тела различны
                    encoded_obs_with_actions.append(self.obs_action_encoders[other_body_part](obs_with_actions))
            encoded_obs_with_actions = torch.stack(encoded_obs_with_actions, dim=1)
            self_attn_inputs.append(encoded_obs_with_actions)

        # собираем все результаты работы энкодеров в один тензор и отправляем на вход механизма внимания
        encoded_entity = torch.cat(self_attn_inputs, dim=1)
        encoded_state = self.self_attn(encoded_entity)

        # TODO: здесь нужно добавить количество агентов, на первом этапе не делаем
        # результаты работы механизма внимания прогоняем через полносвязную сеть
        # для выичсления результатов работы модели критика для нескольких агентов
        encoding = self.linear_encoder(encoded_state)

        return encoding
