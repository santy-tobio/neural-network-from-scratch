"""Experiment configuration classes and predefined configs"""

from .base_config import (
    EarlyStoppingConfig,
    ExperimentConfig,
    OptimizerConfig,
    RegularizerConfig,
    SchedulerConfig,
)
from .predefined import (
    ADAM_CONFIG,
    CV_CONFIG,
    M0_CONFIG,
    M1_CONFIG,
    M2_CONFIG,
    M3_CONFIG,
)

__all__ = [
    # Base config classes
    "ExperimentConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "RegularizerConfig",
    "EarlyStoppingConfig",
    # Predefined experiments
    "M0_CONFIG",
    "M1_CONFIG",
    "M2_CONFIG",
    "M3_CONFIG",
    "ADAM_CONFIG",
    "CV_CONFIG",
]
