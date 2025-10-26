from enum import Enum

from .adam import Adam
from .base_optimizer import BaseOptimizer
from .sgd import SGD

__all__ = ["BaseOptimizer", "SGD", "Adam", "OptimizerType"]


class OptimizerType(Enum):
    SGD = "sgd"
    ADAM = "adam"
