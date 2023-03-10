class BodyPartProperties:
    def __init__(self, hidden_dim, output_dim, obs_space_idxs, action_space_idxs):
        """
        Свойства модели части тела агента Walker
        :param hidden_dim: размер скрытого полносвязного слоя модели актора
        :param output_dim: количество выходов модели актора
        (количество действий, по которым данной части тела нужно принять решение)
        :param obs_space_idxs: индексы пространства наблюдений для данной части тела агента
        :param action_space_idxs: индексы пространства действий для данной части клиента
        """
        self.input_dim = len(obs_space_idxs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.obs_space_idxs = obs_space_idxs
        self.action_space_idxs = action_space_idxs


class WalkerBody:
    """
    Класс свойств агента Walker.
    Представляет собой результат декомпозиции пространства наблюдений и пространства действий агента.
    """
    def __init__(self):
        self.body = dict()

        # полные размеры пространств наблюдений и действий
        self.obs_size = 243
        self.actions_size = 39

        range_list = list(range(self.obs_size))

        # Индексы общей части для всех частей тела.
        # Первые 18 индексов несут информацию о позиции, скорости и направлении движения агента,
        # а так же о позиции целеуказателя, в сторону которого должен двигаться агент.
        # Дополнительно индексы пространства наблюдений для трех частей тела агента,
        # которые не требуют действий (hips, handL, handR). Эту информацию тоже предоставим в качестве
        # наблюдений для каждой части тела.
        self.common_obs_space_idxs = range_list[:18] + range_list[18:28] + range_list[193:203] + range_list[233:243]

        # далее все части тела агента, для которых требуются действия
        self.body['head'] = BodyPartProperties(hidden_dim=32,
                                               output_dim=3,
                                               obs_space_idxs=self.common_obs_space_idxs + range_list[58:73],
                                               action_space_idxs=[24, 25, 28])

        self.body['spine'] = BodyPartProperties(hidden_dim=64,
                                                output_dim=4,
                                                obs_space_idxs=self.common_obs_space_idxs + range_list[43:58],
                                                action_space_idxs=[3, 4, 5, 27])

        self.body['chest'] = BodyPartProperties(hidden_dim=64,
                                                output_dim=4,
                                                obs_space_idxs=self.common_obs_space_idxs + range_list[28:43],
                                                action_space_idxs=[0, 1, 2, 26])

        self.body['thighL'] = BodyPartProperties(hidden_dim=32,
                                                 output_dim=3,
                                                 obs_space_idxs=self.common_obs_space_idxs + range_list[73:88],
                                                 action_space_idxs=[6, 7, 29])

        self.body['thighR'] = BodyPartProperties(hidden_dim=32,
                                                 output_dim=3,
                                                 obs_space_idxs=self.common_obs_space_idxs + range_list[118:133],
                                                 action_space_idxs=[8, 9, 32])

        self.body['shinL'] = BodyPartProperties(hidden_dim=16,
                                                output_dim=2,
                                                obs_space_idxs=self.common_obs_space_idxs + range_list[88:103],
                                                action_space_idxs=[10, 30])

        self.body['shinR'] = BodyPartProperties(hidden_dim=16,
                                                output_dim=2,
                                                obs_space_idxs=self.common_obs_space_idxs + range_list[133:148],
                                                action_space_idxs=[11, 33])

        self.body['footR'] = BodyPartProperties(hidden_dim=64,
                                                output_dim=4,
                                                obs_space_idxs=self.common_obs_space_idxs + range_list[148:163],
                                                action_space_idxs=[12, 13, 14, 34])

        self.body['footL'] = BodyPartProperties(hidden_dim=64,
                                                output_dim=4,
                                                obs_space_idxs=self.common_obs_space_idxs + range_list[103:118],
                                                action_space_idxs=[15, 16, 17, 31])

        self.body['armL'] = BodyPartProperties(hidden_dim=32,
                                               output_dim=3,
                                               obs_space_idxs=self.common_obs_space_idxs + range_list[163:178],
                                               action_space_idxs=[18, 19, 35])

        self.body['armR'] = BodyPartProperties(hidden_dim=32,
                                               output_dim=3,
                                               obs_space_idxs=self.common_obs_space_idxs + range_list[203:218],
                                               action_space_idxs=[20, 21, 37])

        self.body['forearmL'] = BodyPartProperties(hidden_dim=16,
                                                   output_dim=2,
                                                   obs_space_idxs=self.common_obs_space_idxs + range_list[178:193],
                                                   action_space_idxs=[22, 36])

        self.body['forearmR'] = BodyPartProperties(hidden_dim=16,
                                                   output_dim=2,
                                                   obs_space_idxs=self.common_obs_space_idxs + range_list[218:233],
                                                   action_space_idxs=[23, 38])

        self.body_parts_count = len(self.body)
