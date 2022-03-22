from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import supersuit as ss
import torch
from torch import nn
from ray import shutdown

from env import env_creator
from model import CustomVisionNetwork, get_dqn


if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN
    

    # env = ss.color_reduction_v0(env, mode='B')
    # env = ss.dtype_v0(env, 'float32')
    # env = ss.resize_v0(env, x_size=84, y_size=84)
    # env = ss.frame_stack_v1(env, 3)
    # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)

    env = env_creator()
    register_env("battlev3", lambda _: env)
    # model = CustomVisionNetwork(env.observation_space, env.action_space, 256, {}, "custom_vision_network")
    ModelCatalog.register_custom_model("custom_model", CustomVisionNetwork)

    tune.run(
        "APEX",
        stop={"episodes_total": 60000},
        checkpoint_freq=10,
        config={
            # Enviroment specific.
            "env": "battlev3",
            # Model
            "model": {
                # "dim": 13, 
                # "conv_filters": [[32, [3, 3], 1], [32, [3, 3], 1]],
                # "post_fcnet_hiddens": []
                "custom_model": "custom_model"
            },
            # General
            "framework": "torch",
            "num_gpus": 1,
            "num_workers": 2,
            "num_envs_per_worker": 8,
            "learning_starts": 1000,
            "buffer_size": int(1e5),
            "compress_observations": True,
            "rollout_fragment_length": 20,
            "train_batch_size": 512,
            "gamma": 0.99,
            "n_step": 3,
            "lr": 0.0001,
            "prioritized_replay_alpha": 0.5,
            "final_prioritized_replay_beta": 1.0,
            "target_network_update_freq": 50000,
            "timesteps_per_iteration": 25000,
            # Method specific.
            "multiagent": {
                # We only have one policy (calling it "shared").
                # Class, obs/act-spaces, and config will be derived
                # automatically.
                "policies": {"shared_policy"},
                # Always use "shared" policy.
                "policy_mapping_fn": (
                    lambda agent_id, episode, **kwargs: "shared_policy"
                ),
            },
        },
    )
