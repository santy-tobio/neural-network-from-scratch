"""Training utilities including trainer, schedulers, regularizers, and early stopping."""

from enum import Enum

from .early_stopping import EarlyStopping
from .lr_schedulers import (
    ExponentialScheduler,
    LinearScheduler,
    LRScheduler,
)
from .regularizers import L2Regularizer
from .trainer import Trainer

__all__ = [
    "Trainer",
    "LRScheduler",
    "LinearScheduler",
    "ExponentialScheduler",
    "L2Regularizer",
    "EarlyStopping",
    "SchedulerType",
]


class SchedulerType(Enum):
    """Types of learning rate schedulers available."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    COSINE_ANNEALING = "cosine_annealing"
