
class BodyPartProperties:
    def __init__(self, input_dim, hidden_dim, output_dim, obs_space_idxs, action_space_idxs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.obs_space_idxs = obs_space_idxs
        self.action_space_idxs = action_space_idxs


class WalkerBody:
    def __init__(self):
        self.body = dict()

        self.obs_size = 243
        self.actions_size = 39

        self.body['head'] = BodyPartProperties(input_dim=self.obs_size,
                                               hidden_dim=32,
                                               output_dim=3,
                                               obs_space_idxs=slice(58, 73),
                                               action_space_idxs=[24, 25, 28])

        self.body['spine'] = BodyPartProperties(input_dim=self.obs_size,
                                                hidden_dim=64,
                                                output_dim=4,
                                                obs_space_idxs=slice(43, 58),
                                                action_space_idxs=[3, 4, 5, 27])

        self.body['chest'] = BodyPartProperties(input_dim=self.obs_size,
                                                hidden_dim=64,
                                                output_dim=4,
                                                obs_space_idxs=slice(28, 43),
                                                action_space_idxs=[0, 1, 2, 26])

        self.body['thighL'] = BodyPartProperties(input_dim=self.obs_size,
                                                 hidden_dim=32,
                                                 output_dim=3,
                                                 obs_space_idxs=slice(73, 88),
                                                 action_space_idxs=[6, 7, 29])

        self.body['thighR'] = BodyPartProperties(input_dim=self.obs_size,
                                                 hidden_dim=32,
                                                 output_dim=3,
                                                 obs_space_idxs=slice(118, 133),
                                                 action_space_idxs=[8, 9, 32])

        self.body['shinL'] = BodyPartProperties(input_dim=self.obs_size,
                                                hidden_dim=16,
                                                output_dim=2,
                                                obs_space_idxs=slice(88, 103),
                                                action_space_idxs=[10, 30])

        self.body['shinR'] = BodyPartProperties(input_dim=self.obs_size,
                                                hidden_dim=16,
                                                output_dim=2,
                                                obs_space_idxs=slice(133, 148),
                                                action_space_idxs=[11, 33])

        self.body['footR'] = BodyPartProperties(input_dim=self.obs_size,
                                                hidden_dim=64,
                                                output_dim=4,
                                                obs_space_idxs=slice(148, 163),
                                                action_space_idxs=[12, 13, 14, 34])

        self.body['footL'] = BodyPartProperties(input_dim=self.obs_size,
                                                hidden_dim=64,
                                                output_dim=4,
                                                obs_space_idxs=slice(103, 118),
                                                action_space_idxs=[15, 16, 17, 31])

        self.body['armL'] = BodyPartProperties(input_dim=self.obs_size,
                                               hidden_dim=32,
                                               output_dim=3,
                                               obs_space_idxs=slice(163, 178),
                                               action_space_idxs=[18, 19, 35])

        self.body['armR'] = BodyPartProperties(input_dim=self.obs_size,
                                               hidden_dim=32,
                                               output_dim=3,
                                               obs_space_idxs=slice(203, 218),
                                               action_space_idxs=[20, 21, 37])

        self.body['forearmL'] = BodyPartProperties(input_dim=self.obs_size,
                                                   hidden_dim=16,
                                                   output_dim=2,
                                                   obs_space_idxs=slice(178, 193),
                                                   action_space_idxs=[22, 36])

        self.body['forearmR'] = BodyPartProperties(input_dim=self.obs_size,
                                                   hidden_dim=16,
                                                   output_dim=2,
                                                   obs_space_idxs=slice(218, 233),
                                                   action_space_idxs=[23, 38])

        self.body_parts_count = len(self.body)

        range_list = list(range(self.obs_size))
        # общая часть + hips + handL + handR
        self.common_obs_space_idxs = range_list[:18] + range_list[18:28] + range_list[193:203] + range_list[233:243]
