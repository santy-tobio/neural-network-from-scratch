from .config import (
    EarlyStoppingConfig,
    RegularizerConfig,
    SchedulerConfig,
    SchedulerType,
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

# Try importing PyTorch trainer (optional dependency)
try:
    from .trainer_pytorch import TrainerPyTorch
    _PYTORCH_AVAILABLE = True
except ImportError:
    TrainerPyTorch = None
    _PYTORCH_AVAILABLE = False

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

# Add PyTorch exports if available
if _PYTORCH_AVAILABLE:
    __all__.append("TrainerPyTorch")
