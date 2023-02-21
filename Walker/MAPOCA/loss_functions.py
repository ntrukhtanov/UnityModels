import torch


def calc_policy_loss(advantages, log_probs, old_log_probs, mask, epsilon):
    """
    Функция расчета значения функции потерь для актора.
    :param advantages: Значения функции преимущества.
    :param log_probs: Значения логарифмов вероятностей действий, рассчитанных на основе текущей модели актора.
    :param old_log_probs: Значения логарифмов вероятностей действий, рассчитанных на основе старой модели актора.
    :param mask: Маска для исключения из расчета данных от сломанных частей тела агента
    :param epsilon: Значение константы для обрезки до граничных значений с целью
    ограничить слишком большие значения функции потерь
    :return: Значение функции потерь
    """
    r_theta = torch.exp(log_probs - old_log_probs)
    p_opt_a = r_theta * advantages
    p_opt_b = torch.clamp(r_theta, 1.0 - epsilon, 1.0 + epsilon) * advantages

    # берем со знаком минус, т.к. у используем алгоритм градиента стратегии и нам нужен градиентный подъем,
    # а оптимизатор выполняет градиентный спуск
    policy_loss = torch.sum(torch.min(p_opt_a, p_opt_b), dim=1, keepdim=True)
    policy_loss = -1 * torch.sum(mask * policy_loss) / mask.sum()
    return policy_loss


def calc_value_loss(values, old_values, returns, mask, epsilon):
    """
    Функция расчета значения функции потерь для критика
    :param values: Значения полученные на выходе модели критика, рассчитанные на основе текущей модели
    :param old_values: Значения полученные на выходе модели критика, рассчитанные на основе старой модели
    :param returns: Дисконтированные значения вознаграждений
    :param mask: Маска для исключения из расчета данных от сломанных частей тела агента
    :param epsilon: Значение константы для обрезки до граничных значений с целью
    ограничить слишком большие значения функции потерь
    :return: Значение функции потерь
    """
    clipped_value_estimate = old_values + torch.clamp(
        values - old_values, -1 * epsilon, epsilon
    )
    v_opt_a = (returns - values) ** 2
    v_opt_b = (returns - clipped_value_estimate) ** 2
    loss = torch.max(v_opt_a, v_opt_b)
    loss = torch.sum(mask * loss) / mask.sum()
    return loss


def calc_returns(rewards, dones, values, gamma, lam, value_next):
    """
    Функция расчета дисконтированного вознаграждения методом GAE (Обобщенная оценка преимущества).
    :param rewards: Вознаграждения.
    :param dones: Признаки завершения эпизода.
    :param values: Значения модели критика
    :param gamma: Коэффициент дисконтирования вознаграждений
    :param lam: Вес оценки модели критика относительно вознаграждения
    :param value_next: Значения модели критика на последнем шаге
    :return:
    """
    returns = torch.zeros_like(rewards)
    # вычисляем последние значения на основе value_next, а затем двигаемся от конца к началу
    returns[-1] = rewards[-1] + gamma * value_next * (1 - dones[-1])
    for t in reversed(range(0, rewards.shape[0] - 1)):
        returns[t] = (
                gamma * lam * returns[t + 1] * (1 - dones[t])
                + rewards[t]
                + (1 - lam) * gamma * values[t + 1] * (1 - dones[t])
        )
    return returns
