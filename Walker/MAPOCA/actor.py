import torch
from torch import nn
import math

from torch.distributions.normal import Normal

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
        # которые должна определить модель для данной части тела агента
        self.actor_body = nn.Sequential(
            nn.Linear(input_dim, body_part_properties.hidden_dim),
            nn.ReLU(),
            nn.Linear(body_part_properties.hidden_dim, body_part_properties.output_dim)
        )

        # обучаемый параметр логарифм дисперсии, нужен для вычисления значений действий, пояснение в функции forward
        self.log_sigma = nn.Parameter(
            torch.zeros(1, body_part_properties.output_dim, requires_grad=True)
        )

        # параметр для хранения максимальных значений для каждой фичи с целью нормализации данных
        self.norm_values = nn.Parameter(
            torch.ones(1, input_dim, requires_grad=False)
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

        # если параметр norm_values содержит меньшее значение, то заменим на большее
        self.norm_values.data = torch.max(self.norm_values.data, max_values)

        # выполним нормализацию
        input_data = input_data / self.norm_values.data

        return input_data


    def forward(self, input_data):
        """
        Функция прямого прохождения модели с вычислением статистики
        :param input_data: входные данные о наблюдениях соответствующей части тела агента
        :return: возвращает:
        actions: значения действий
        log_prob: значения логарифмов вероятностей действий
        entropy: энтропия
        """
        # выполним нормализацию данных
        input_data_norm = self.normalize_input(input_data)

        # результаты работы сети будем считать средним значением для нормального распределения
        mean = self.actor_body(input_data_norm)

        # Здесь важный механизм выбора между исследованием среды и применением модели.
        # Вносим в вычисленные значения шум с дисперсией,
        # вычисленной на основании обучаемого параметра логарифма дисперсии.
        # Это позволяет алгоритму больше исследовать окружающую среду в начале обучения.
        # А в будущем позволяет чаще выбирать значения действий, приносящие большее вознаграждение,
        # по мере уменьшения дисперсии (по мере обучения и роста уверенности модели).
        std = torch.exp(self.log_sigma)

        # инициализируем нормальное распределение
        pd = Normal(loc=mean, scale=std)

        # семплируем действия согласно параметрам распределения
        actions = pd.sample()

        # Вычисляем логарифмы вероятностей действий
        log_prob = pd.log_prob(actions)

        return actions, log_prob

    def evaluate_actions(self, input_data, actions):
        """
        Функция оценки действий с помощью текущей модели. Применяется для рассчета функции потерь модели актора.
        :param input_data: данные о наблюдениях для текущей части тела агента
        :param actions: действия, которые были применены к данной части тела агента предыдущей моделью
        :return: возвращает значения логарфимов вероятностей действий и энтропию
        """

        # выполним нормализацию данных
        input_data_norm = self.normalize_input(input_data)

        mean = self.actor_body(input_data_norm)
        std = torch.exp(self.log_sigma)

        # инициализируем нормальное распределение
        pd = Normal(loc=mean, scale=std)

        # Вычисляем логарифмы вероятностей действий
        log_prob = pd.log_prob(actions)

        # Вычисляем энтропию.
        # Энтропия растет по мере роста дисперсии и со знаком минус участвует в расчете суммарной функции потерь.
        # Таким образом, оптмизатору выгодно увеличивать энтропию и тем самым повысить шум, накладываемый на действия,
        # и тем самым стимулировать исследование.
        # С другой стороны энтропия в расчете функции потерь присутствует с коэффициентом, который не позволяет,
        # сделать оптимизацию за счет энтропии более выгодной, чем за счет оптимизации стратегии.
        entropy = pd.entropy()

        return log_prob, entropy
