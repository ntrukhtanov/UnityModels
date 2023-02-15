import torch
from torch import nn

from linear_encoder import LinearEncoder


class EntityEmbedding(nn.Module):
    def __init__(self, obs_space_size, actions_space_size, embedding_size):
        """
        Вспомогательная модель для вычисления эмбедингов
        :param entity_size: Размерность входных данный
        :param embedding_size: Размерность эмбедингов
        """
        super().__init__()

        self.obs_space_size = obs_space_size
        self.actions_space_size = actions_space_size

        self.embedding = LinearEncoder(obs_space_size + actions_space_size, 1, embedding_size)

        # параметр для хранения максимальных значений для каждой фичи с целью нормализации данных
        self.norm_values = nn.Parameter(
            torch.ones(1, obs_space_size + actions_space_size, requires_grad=False)
        )
        self.norm_values.requires_grad_(False)

    def normalize_input(self, input_data):
        """
        Функция нормализации данных
        :param input_data: Входные данные для нормализации
        :return: Нормализованные данные
        """

        # Определим максимальные по модулю значения для каждой фичи в батче
        max_values = torch.max(torch.abs(input_data), dim=0).values

        # отключим нормализацию для действий, потому что это обучаемые значения
        # и их макимум будет "плавать" по мере обучения
        max_values[self.obs_space_size:] = 1.

        # если параметр norm_values содержит меньшее значение, то заменим на большее
        self.norm_values.data = torch.max(self.norm_values.data, max_values)

        # выполним нормализацию
        input_data = input_data / self.norm_values.data

        return input_data

    def forward(self, input_data):
        input_data = self.normalize_input(input_data)
        return self.embedding(input_data)
