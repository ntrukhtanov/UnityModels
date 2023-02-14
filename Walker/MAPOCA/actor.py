import torch
from torch import nn
import math

EPSILON = 1e-7  # Small value to avoid divide by zero


class ActorModel(nn.Module):
    def __init__(self, body_part, body_part_properties, input_dim):
        """
        Модель актора, отдельная для каждой части тела агента
        :param body_part: наименование части тела
        :param body_part_properties: свойства части тела
        :param input_dim: размерность пространства наблюдений для части тела (для актора всегда 243)
        """
        super().__init__()
        self.body_part = body_part

        # линейный слой
        # на вход получаем пространство наблюдений части тела агента N x 243, где N - размер батча
        # на выходе получаем тензор размерности N x A, где A - количество непрерывных действий,
        # которые должна определить модель для данной части тела
        self.actor_body = nn.Sequential(
            nn.Linear(input_dim, body_part_properties.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(body_part_properties.hidden_dim, body_part_properties.output_dim)
        )

        # обучаемый параметр логарифм дисперсии, нужен для вычисления значений действий, пояснение в функции forward
        self.log_sigma = nn.Parameter(
            torch.zeros(1, body_part_properties.output_dim, requires_grad=True)
        )

    def forward(self, input_data):
        """
        Функция прямого прохождения модели
        :param input_data: входные данные о наблюдениях соответствующей части тела агента
        :return: возвращает значения действий
        """

        # модель вычисляет средние значений действий
        mean = self.actor_body(input_data)

        # Здесь важный механизм выбора между исследованием среды и применением модели.
        # Вносим в вычисленные значения шум с дисперсией,
        # вычисленной на основании обучаемого параметра логарифма дисперсии.
        # Это позволяет алгоритму больше исследовать окружающую среду в начале обучения.
        # А в будущем позволяет чаще выбирать значения действий, приносящие большее вознаграждение,
        # по мере уменьшения дисперсии (по мере обучения и роста уверенности модели).
        std = torch.exp(self.log_sigma)
        rand_shift = torch.randn_like(mean) * std
        actions = mean + rand_shift

        return actions

    def forward_with_stat(self, input_data):
        """
        Функция прямого прохождения модели с вычислением статистики
        :param input_data: входные данные о наблюдениях соответствующей части тела агента
        :return: возвращает:
        actions: значения действий
        log_prob: значения логарифмов вероятностей действий
        entropy: энтропия
        """
        mean = self.actor_body(input_data)

        std = torch.exp(self.log_sigma)
        rand_shift = torch.randn_like(mean) * std
        actions = mean + rand_shift

        log_prob, entropy = self.get_stats(actions, mean, std)

        return actions, log_prob, entropy

    def get_stats(self, actions, mean, std):
        """
        Функция расчета статистики
        :param actions: Значения действий с учетом внесенног шума
        :param mean: Значения действий без внесенного шума
        :param std: Дисперсия шума
        :return: Возвращает логарифмы вероятностей действий и энтропию
        """

        # Вычисляем логарифмы вероятностей действий по формуле,
        # взятой из исходных кодов к алгоритму https://arxiv.org/pdf/2111.05992.pdf
        var = std ** 2
        log_scale_var = torch.log(std + EPSILON)

        log_prob = -((actions - mean) ** 2) / (2 * var + EPSILON) - log_scale_var - math.log(math.sqrt(2 * math.pi))

        # вычисляем энтропию по формуле, взятой из исходных кодов к алгоритму https://arxiv.org/pdf/2111.05992.pdf
        # энтропия растет по мере роста дисперсии и со знаком минус участвует в расчете суммарной функции потерь
        # таким образом
        # TODO: нужно здесь закончить объяснение
        entropy = torch.mean(0.5 * torch.log(2 * math.pi * math.e * std ** 2 + EPSILON), dim=1, keepdim=True)

        return log_prob, entropy

    def evaluate_actions(self, input_data, actions):
        mean = self.actor_body(input_data)
        std = torch.exp(self.log_sigma)

        log_prob, entropy = self.get_stats(actions, mean, std)

        return log_prob, entropy
