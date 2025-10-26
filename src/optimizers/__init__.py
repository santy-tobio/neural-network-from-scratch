from enum import Enum
from .base_optimizer import BaseOptimizer
from .sgd import SGD
from .adam import Adam

__all__ = ["BaseOptimizer", "SGD", "Adam", "OptimizerType"]


class OptimizerType(Enum):
    SGD = "sgd"
    ADAM = "adam"
