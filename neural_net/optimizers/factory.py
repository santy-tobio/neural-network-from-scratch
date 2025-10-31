"""Factory functions for creating optimizers from configurations."""

from .base import BaseOptimizer
from .config import OptimizerConfig


def create_optimizer(config: OptimizerConfig) -> BaseOptimizer:
    """Create an optimizer instance from configuration."""
    optimizer_class = config.type.value
    return optimizer_class.from_config(config)
