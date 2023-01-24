from dataclasses import dataclass, field
from typing import List, Callable
import functools

from ray.rllib.models.experimental.base import ModelConfig, Model
from ray.rllib.utils.annotations import DeveloperAPI


@DeveloperAPI
def _framework_implemented(torch: bool = True, tf: bool = True):
    """Decorator to check if a model was implemented in a framework.

    Args:
        torch: Whether we can build this model with torch.
        tf: Whether we can build this model with tf.

    Returns:
        The decorated function.

    Raises:
        ValueError: If the framework is not available to build.
    """
    accepted = []
    if torch:
        accepted.append("torch")
    if tf:
        accepted.append("tf")
        accepted.append("tf2")

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def checked_build(self, framework, **kwargs):
            if framework not in accepted:
                raise ValueError(f"Framework {framework} not supported.")
            return fn(self, framework, **kwargs)

        return checked_build

    return decorator


@dataclass
class FCConfig(ModelConfig):
    """Configuration for a fully connected network.

    Attributes:
        input_dim: The input dimension of the network. It cannot be None.
        hidden_layers: The sizes of the hidden layers.
        activation: The activation function to use after each layer (except for the
            output).
        output_activation: The activation function to use for the output layer.
    """

    input_dim: int = None
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "ReLU"
    output_activation: str = "ReLU"

    @_framework_implemented()
    def build(self, framework: str = "torch") -> Model:
        if framework == "torch":
            from ray.rllib.models.experimental.torch.fcmodel import FCModel
        else:
            from ray.rllib.models.experimental.tf.fcmodel import FCModel
        return FCModel(self)


@dataclass
class FCEncoderConfig(FCConfig):
    @_framework_implemented()
    def build(self, framework: str = "torch"):
        if framework == "torch":
            from ray.rllib.models.experimental.torch.encoder import FCEncoder
        else:
            from ray.rllib.models.experimental.tf.encoder import FCEncoder
        return FCEncoder(self)


@dataclass
class LSTMEncoderConfig(ModelConfig):
    input_dim: int = None
    hidden_dim: int = None
    num_layers: int = None
    batch_first: bool = True

    @_framework_implemented(tf=False)
    def build(self, framework: str = "torch"):
        if framework == "torch":
            from rllib.models.experimental.torch.encoder import LSTMEncoder

        return LSTMEncoder(self)


@dataclass
class IdentityConfig(ModelConfig):
    """Configuration for an identity encoder."""

    @_framework_implemented()
    def build(self, framework: str = "torch"):
        if framework == "torch":
            from rllib.models.experimental.torch.encoder import IdentityEncoder
        else:
            from rllib.models.experimental.tf.encoder import IdentityEncoder

        return IdentityEncoder(self)
