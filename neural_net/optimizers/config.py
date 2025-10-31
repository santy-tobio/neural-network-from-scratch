from dataclasses import dataclass

from pyparsing import Enum

from .adam import Adam
from .sgd import SGD


class OptimizerType(Enum):
    """Types of optimizers available."""

    SGD = SGD
    ADAM = Adam


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    type: OptimizerType = OptimizerType.SGD
    learning_rate: float = 0.001

    momentum: float = 0.0
    nesterov: bool = False

    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self):
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert 0 <= self.momentum < 1, "Momentum must be in [0, 1)"
        assert 0 < self.beta1 < 1, "Beta1 must be in (0, 1)"
        assert 0 < self.beta2 < 1, "Beta2 must be in (0, 1)"

    def __repr__(self) -> str:
        """Clean representation for logging."""
        if self.type == OptimizerType.SGD:
            return (
                f"OptimizerConfig(type=SGD, lr={self.learning_rate}, "
                f"momentum={self.momentum}, nesterov={self.nesterov})"
            )
        elif self.type == OptimizerType.ADAM:
            return (
                f"OptimizerConfig(type=ADAM, lr={self.learning_rate}, "
                f"beta1={self.beta1}, beta2={self.beta2})"
            )
        return f"OptimizerConfig(type={self.type.name}, lr={self.learning_rate})"
