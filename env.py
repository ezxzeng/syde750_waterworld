
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.magent import battle_v3
from pettingzoo.sisl import waterworld_v3

def env_creator():
    env = battle_v3.env(map_size=45, minimap_mode=False, step_reward=-0.005,
            dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
            max_cycles=1000, extra_features=False)
    # env = waterworld_v3.env()
    return PettingZooEnv(env)