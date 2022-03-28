from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from waterworld import waterworld as custom_waterworld

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl

if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    register_env(
        "custom_waterworld_1",
        lambda _: PettingZooEnv(custom_waterworld.env(n_sensors=1)),
    )

    tune.run(
        "APEX_DDPG",
        stop={"episodes_total": 60000},
        checkpoint_freq=10,
        config={
            # Enviroment specific.
            "env": "custom_waterworld_1",
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
