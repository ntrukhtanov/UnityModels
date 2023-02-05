from mlagents_envs.environment import UnityEnvironment
import numpy as np


def run():
    env = UnityEnvironment("./Unity/DungeonEscape", no_graphics=True)
    env.reset()
    behavior_name = None
    for behavior_name in env.behavior_specs:
        print(behavior_name)
    action_spec = env.behavior_specs[behavior_name].action_spec
    observation_specs = env.behavior_specs[behavior_name].observation_specs
    while True:
        ds, ts = env.get_steps(behavior_name)
        if ds.action_mask is not None:
            n_agents = ds.agent_id.shape[0]
            print(f'n_agents: {n_agents}, rewards: {ds.reward}')
            random_actions = action_spec.random_action(n_agents)
            env.set_actions(behavior_name, random_actions)
        env.step()


if __name__ == '__main__':
    run()