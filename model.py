import numpy as np

import torch
from torch import nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.visionnet import VisionNetwork
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)

class CustomVisionNetwork(VisionNetwork):


    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        activation = self.model_config.get("conv_activation")

        hidden_size = [512, 256]

        filters = [[32, [3, 3], 1], [32, [3, 3], 1]]

        layers = []
        (w, h, in_channels) = obs_space.shape
        in_size = [w, h]

        for out_channels, kernel, stride in filters:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        in_size = [
            np.ceil((in_size[0] - kernel[0]) / stride),
            np.ceil((in_size[1] - kernel[1]) / stride),
        ]
        padding, _ = same_padding(in_size, [1, 1], [1, 1])
        layers.append(nn.Flatten())
        in_size = out_channels
        # Add (optional) post-fc-stack after last Conv2D layer.
        for i, out_size in enumerate(hidden_size):
            layers.append(
                SlimFC(
                    in_size=in_size,
                    out_size=out_size,
                    activation_fn="relu"
                    if i < len(hidden_size) - 1
                    else None,
                    initializer=normc_initializer(1.0),
                )
            )
            in_size = out_size
        # Last layer is logits layer.
        self._logits = layers.pop()

        self._convs = nn.Sequential(*layers)

        self._value_branch_separate = self._value_branch = None
        self._value_branch = SlimFC(
            out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
        )

        self._features = None


def get_dqn(obs_space, action_space, num_outputs, model_config, name):
    return ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=action_space.n,
        model_config=MODEL_DEFAULTS,
        framework="torch",
        # Providing the `model_interface` arg will make the factory
        # wrap the chosen default model with our new model API class
        # (DuelingQModel). This way, both `forward` and `get_q_values`
        # are available in the returned class.
        model_interface=DQNTorchModel,
        name=name,
    )