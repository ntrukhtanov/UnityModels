import torch
from torch import nn

from multi_agent_network_body import MultiAgentNetworkBody

HIDDEN_SIZE = 128


class CriticModel(nn.Module):
    def __init__(self, walker_body):
        """
        Модель критика
        :param walker_body: Описание структуры агента Walker
        """
        super().__init__()

        # инициализируем модель критика для нескольких агентов
        self.multiagent_network = MultiAgentNetworkBody(walker_body, HIDDEN_SIZE)

        # Инициализируем полносвязный выходной слой с выходом размерностью Nx1, где N - размер батча.
        # Полученные значения будут использоваться для вычисления функции потерь.
        # Размер входного слоя определяется как сумма размера выхода модели критика для нескольких агентов
        # и значения количества агентов. В нашем случае действующих частей тела.
        value_encoding_size = HIDDEN_SIZE # + 1
        self.value_heads = nn.Linear(value_encoding_size, 1)

    def forward(self, encoding):
        """
        Функция прямого прохождения модели. Используется для вычисления выходных значений модели критика.
        :param encoding: Результаты прямого прохождения модели критика для нескольких агентов
        :return: Выходные значения модели критика
        """
        return self.value_heads(encoding)

    def critic_common(self, batch):
        """
        Функция прямого прохождения модели критика на основании наблюдений всех частей тела агента,
        без учета их действий.
        :param batch: Батч с данными
        :return: Выходные значения модели критика
        """
        encoding = self.multiagent_network(batch=batch, body_part=None)

        value_outputs = self.forward(encoding)

        return value_outputs

    def critic_body_part(self, batch, body_part):
        """
        Функция прямого прохождения модели критика на основании наблюдений всех частей тела агента,
        с учетом действий частей тела отличных от текущего.
        :param batch: Батч с данными
        :param body_part: Текущая часть тела агента
        :return: Выходные значения модели критика
        """
        encoding = self.multiagent_network(batch=batch, body_part=body_part)

        value_outputs = self.forward(encoding)

        return value_outputs
