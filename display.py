from pettingzoo.sisl import waterworld_v3

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

from waterworld.waterworld import env as custom_waterworld

import ray
import pickle5 as pickle
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from PIL import Image
import numpy as np
import os
import argparse
from pathlib import Path

from ray.rllib.agents.ddpg import ApexDDPGTrainer

import glob


def display_results(checkpoint_path, save_path, env_name, env):
    params_path = Path(checkpoint_path).parent.parent/"params.pkl"
    with open(params_path, "rb") as f:
        config = pickle.load(f)
        # num_workers not needed since we are not training
        del config['num_workers']
        del config['num_gpus']

    ray.init(num_cpus=8, num_gpus=1)
    ApexDDPGagent = ApexDDPGTrainer(env=env_name, config=config)
    ApexDDPGagent.restore(checkpoint_path)

    reward_sum = 0
    frame_list = []
    i = 0
    env.reset()

    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        reward_sum += reward
        if done:
            action = None
        else:
            action, _, _ = ApexDDPGagent.get_policy("shared_policy").compute_single_action(observation)

        env.step(action)
        i += 1
        if i % (len(env.possible_agents)+1) == 0:
            frame_list.append(Image.fromarray(env.render(mode='rgb_array')))
    env.close()


    print(reward_sum)
    frame_list[0].save(save_path, save_all=True, append_images=frame_list[1:], duration=3, loop=0)

    ray.shutdown()


def waterworld_display(checkpoint_path):
    save_path = Path(checkpoint_path).parent/"out.gif"
    
    env = waterworld_v3.env()
    register_env(
        "waterworld",
        lambda _: PettingZooEnv(waterworld_v3.env()),
    )

    display_results(checkpoint_path, save_path, "waterworld", env)


def custom_waterworld_display(checkpoint_path, *args, **kwargs):
    save_path = Path(checkpoint_path).parent/"out.gif"
    
    env = custom_waterworld(*args, **kwargs)
    register_env(
        "waterworld",
        lambda _: PettingZooEnv(custom_waterworld(*args, **kwargs)),
    )

    display_results(checkpoint_path, save_path, "waterworld", env)

if __name__ == "__main__":
    checkpoints = [
        filename for filename in 
        glob.glob("results/APEX_DDPG_waterworld_8bafb_00000_0_2022-03-27_14-40-40/checkpoint_*/*") 
        if "tune_metadata" not in filename
        ]

    for checkpoint in checkpoints:
        waterworld_display(checkpoint)