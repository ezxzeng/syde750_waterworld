from waterworld.waterworld import env as custom_waterworld
from potential_field.potential_field_policy import PotentialFieldPolicy
from utils import get_frames

from pettingzoo.utils import average_total_reward

from multiprocessing import Pool, cpu_count

import tqdm
import numpy as np
from matplotlib import pyplot as plt
import json


n_coop_options = [1, 2]
n_sensor_options = [1, 2, 5, 20, 30]
angle_options = [("randomize_angle",False),("randomize_angle",True),
                ("spin_angle",0),("spin_angle",0.1),("spin_angle",0.5),("spin_angle",1)]
obs_weighting_options=[1, 0.5]
poison_weighting_options=[1, 0.5]
barrier_weighting_options=[1, 0.5]
food_weighting_options=[1, 0.5]

def test_policy(config, rounds=100):
    env = custom_waterworld(**config["env_config"])
    policy = PotentialFieldPolicy(**config["potential_field_config"]).get_movement_vector
    for i in tqdm.tqdm(range(rounds)):
        reward_sum, frame_list = get_frames(env, policy)
        config["rewards"].append(reward_sum)
    env.close()
    
    with open(f"potential_field/test_main/{config['config_index']}.json", "x") as f:
        json.dump(config, f, indent=4)

def get_configs():
    configs = []
    i=0
    for n_coop in n_coop_options:
        for n_sensor in n_sensor_options:
            for angle_config in angle_options:
                configs.append({"env_config":
                            {"n_coop": n_coop,"n_sensors": n_sensor,},
                            "potential_field_config":{
                                "n_sensors": n_sensor,
                                angle_config[0]: angle_config[1],
                            },
                                "rewards": [],
                                "config_index": i
                            })
                i += 1
        for obs_weight in obs_weighting_options:
            for poison_weight in poison_weighting_options:
                for barrier_weight in barrier_weighting_options:
                    for food_weight in food_weighting_options:
                        configs.append({"env_config":
                                    {"n_coop": n_coop,"n_sensors": 30,},
                                    "potential_field_config":{
                                        "n_sensors": 30,
                                        "obs_weight": obs_weight,
                                        "poison_weight": poison_weight,
                                        "barrier_weight": barrier_weight,
                                        "food_weight": food_weight
                                    },
                                        "rewards": [],
                                        "config_index": i
                                    })
                        i += 1
    return configs

def get_main_configs():
    configs = []
    i=0
    for n_coop in n_coop_options:
        for n_sensor in n_sensor_options:
            for angle_config in angle_options:
                configs.append({"env_config":
                            {"n_coop": n_coop,"n_sensors": n_sensor,},
                            "potential_field_config":{
                                "n_sensors": n_sensor,
                                angle_config[0]: angle_config[1],
                            },
                                "rewards": [],
                                "config_index": i
                            })
                i += 1

    return configs


def get_env_configs():
    configs = []
    i=0
    for n_coop in n_coop_options:
        for n_sensor in n_sensor_options:
                configs.append({"env_config":
                            {"n_coop": n_coop,"n_sensors": n_sensor,},
                                "rewards": [],
                                "config_index": i
                            })
                i += 1

    return configs


def test_random_env(config, rounds=100):
    env = custom_waterworld(**config["env_config"])
    action_space = env.action_space("pursuer_0")
    def policy(obs):
        return action_space.sample()

    for i in tqdm.tqdm(range(rounds)):
        reward_sum, frame_list = get_frames(env, policy)
        config["rewards"].append(reward_sum)
    env.close()
    
    with open(f"potential_field/test_random/{config['config_index']}.json", "x") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    configs = get_env_configs()

    with Pool(processes=int(cpu_count() - 2)) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(test_random_env, configs), total=len(configs)):
            pass
