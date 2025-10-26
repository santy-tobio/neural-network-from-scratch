"""Training utilities including trainer, schedulers, regularizers, and early stopping."""

from enum import Enum

from .config import (
    EarlyStoppingConfig,
    RegularizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from .early_stopping import EarlyStopping
from .factory import create_early_stopping, create_regularizer, create_scheduler
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
    # Factories
    "create_scheduler",
    "create_regularizer",
    "create_early_stopping",
    # Configs
    "TrainingConfig",
    "SchedulerConfig",
    "RegularizerConfig",
    "EarlyStoppingConfig",
]

from .lr_schedulers import (
    CosineAnnealingScheduler,
    StepScheduler,
)


class SchedulerType(Enum):
    """Types of learning rate schedulers available."""

    LINEAR = LinearScheduler
    EXPONENTIAL = ExponentialScheduler
    STEP = StepScheduler
    COSINE_ANNEALING = CosineAnnealingScheduler
