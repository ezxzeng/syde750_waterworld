from pettingzoo.sisl import waterworld_v3

from waterworld.waterworld import env as custom_waterworld
from agent.potential_field_policy import PotentialFieldPolicy

from PIL import Image
import numpy as np
import os
import argparse
from pathlib import Path

from utils import get_frames

if __name__ == "__main__":
    n_sensors = 20
    env = custom_waterworld(n_sensors=n_sensors, n_coop=1)
    policy = PotentialFieldPolicy().get_movement_vector
    reward_sum, frame_list = get_frames(env, policy)
    print(reward_sum)
    frame_list[0].save(
        "potential_field.gif",
        save_all=True,
        append_images=frame_list[1:],
        duration=3,
        loop=0,
    )
