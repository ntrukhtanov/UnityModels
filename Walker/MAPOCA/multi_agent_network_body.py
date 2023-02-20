import torch
from torch import nn

from linear_encoder import LinearEncoder
from entity_embedding import EntityEmbedding
from attention import ResidualSelfAttention

EMBEDDING_SIZE = 128


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
            self.obs_encoders[body_part] = EntityEmbedding(body_part_prop.input_dim, 0, EMBEDDING_SIZE)
            self.add_module(f"obs_encoder_{body_part}", self.obs_encoders[body_part])
            # создаем модель эмбединга для текущей части тела с входом для наблюдений + действий
            self.obs_action_encoders[body_part] = EntityEmbedding(
                body_part_prop.input_dim, body_part_prop.output_dim, EMBEDDING_SIZE)
            self.add_module(f"obs_action_encoder_{body_part}", self.obs_action_encoders[body_part])

        # инициализируем модель, реализующую механизм внимания
        self.self_attn = ResidualSelfAttention(EMBEDDING_SIZE)

        # инициализируем модель из полносвязных слоев, которая будет являться выходом данной модели
        self.linear_encoder = LinearEncoder(EMBEDDING_SIZE, 2, hidden_size)

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

        # список, в котором собираем маски для механизма внимания, чтобы исключить сломанные части тела агента
        self_attn_masks = list()

        if body_part is None:
            # вычисляем результаты работы энкодеров на основании наблюдений всех частей тела, без учета действий
            all_encoded_obs = list()
            masks = list()
            for body_part in self.walker_body.body.keys():
                obs = batch[body_part]["obs"]
                # если наблюдение для текущей части тела содержит хотя бы одно значение nan,
                # заполняем для него маску значением True
                mask = torch.isnan(obs)
                mask = mask.sum(dim=1)
                mask = mask > 0
                masks.append(mask)

                # обнулим значения nan для отправки в энкодер
                obs = torch.nan_to_num(obs)

                # Для каждой части тела свой энкодер наблюдений, т.к. наблюдения имеют разную размерность
                # и части тела выполняют разные функции.
                all_encoded_obs.append(self.obs_encoders[body_part](obs))

            # собираем все эмбединги в один тензор и добавляем в список для механизма внимания
            all_encoded_obs = torch.stack(all_encoded_obs, dim=1)
            self_attn_inputs.append(all_encoded_obs)

            # собираем все маски в один тензор и добавляем в список для механизма внимания
            masks = torch.stack(masks, dim=1)
            self_attn_masks.append(masks)
        else:
            # Вычислим результаты работы энкодеров на основании наблюдения текущей части тела (параметр body_part)
            # без учета его действий,
            # и на основании наблюдений всех остальных частей тела, с учетом их действий

            obs = batch[body_part]["obs"]

            # если наблюдение для текущей части тела содержит хотя бы одно значение nan,
            # заполняем для него маску значением True
            mask = torch.isnan(obs)
            mask = mask.sum(dim=1)
            mask = mask > 0

            # обнулим значения nan для отправки в энкодер
            obs = torch.nan_to_num(obs)

            # вычисляем энкодер для текущй части тела и добавляем в список на вход механизма внимания
            encoded_obs = self.obs_encoders[body_part](obs)
            encoded_obs = torch.stack([encoded_obs], dim=1)
            self_attn_inputs.append(encoded_obs)

            # добавляем в список маску для текущей части тела
            masks = torch.stack([mask], dim=1)
            self_attn_masks.append(masks)

            # проходим по всем частям тела кроме текущей и вычисляем энкодеры с учетом их действий
            encoded_obs_with_actions = list()
            masks = list()
            for other_body_part in self.walker_body.body.keys():
                if other_body_part != body_part:
                    obs = batch[other_body_part]["obs"]
                    actions = batch[other_body_part]["actions"]

                    # если наблюдение для текущей части тела содержит хотя бы одно значение nan,
                    # заполняем для него маску значением True
                    obs_mask = torch.isnan(obs)
                    obs_mask = obs_mask.sum(dim=1)
                    obs_mask = obs_mask > 0

                    # обнулим значения nan для отправки в энкодер
                    obs = torch.nan_to_num(obs)

                    # если действия для текущей части тела содержат хотя бы одно значение nan,
                    # заполняем для него маску значением True
                    actions_mask = torch.isnan(actions)
                    actions_mask = actions_mask.sum(dim=1)
                    actions_mask = actions_mask > 0

                    # обнулим значения nan для отправки в энкодер
                    actions = torch.nan_to_num(actions)

                    # объединяем маски логическим или
                    mask = obs_mask | actions_mask

                    # и добавляем в список
                    masks.append(mask)

                    # объединяем наблюдения и действия в один тензор
                    obs_with_actions = torch.cat([obs, actions], dim=1)

                    # для каждой части тела свой отдельный энкодер,
                    # т.к. пространства наблюдений и действий разнородны и функции частей тела различны
                    encoded_obs_with_actions.append(self.obs_action_encoders[other_body_part](obs_with_actions))

            # собираем все эмбединги в один тензор и добавляем в список для механизма внимания
            encoded_obs_with_actions = torch.stack(encoded_obs_with_actions, dim=1)
            self_attn_inputs.append(encoded_obs_with_actions)

            # собираем все маски в один тензор и добавляем в список для механизма внимания
            masks = torch.stack(masks, dim=1)
            self_attn_masks.append(masks)

        # собираем все результаты работы энкодеров в один тензор и отправляем на вход механизма внимания
        encoded_entity = torch.cat(self_attn_inputs, dim=1)
        mask = torch.cat(self_attn_masks, dim=1)
        encoded_state = self.self_attn(encoded_entity, mask)

        # результаты работы механизма внимания прогоняем через полносвязную сеть
        # для выичсления результатов работы модели критика для нескольких агентов
        encoding = self.linear_encoder(encoded_state)

        return encoding
