"""
Neural Network Library

A modular neural network implementation with MLP models,
optimizers (SGD, Adam), training utilities, and evaluation metrics.
"""

__version__ = "0.1.0"

# Import key classes for convenient access
from .data import Dataset, KFoldSplitter
from .evaluation import compute_metrics, evaluate_model
from .factory import create_training_components
from .layers import CESoftmax, Dropout, Layer, LayerType, Linear, Relu
from .models import MLP, BaseMLP, ModelConfig
from .optimizers import SGD, Adam, OptimizerConfig, OptimizerType
from .training import (
    EarlyStoppingConfig,
    RegularizerConfig,
    SchedulerConfig,
    SchedulerType,
    Trainer,
    TrainingConfig,
)

__all__ = [
    # Factory
    "create_training_components",
    # Models
    "MLP",
    "BaseMLP",
    "ModelConfig",
    # Layers
    "Layer",
    "Linear",
    "Relu",
    "CESoftmax",
    "Dropout",
    "LayerType",
    # Optimizers
    "SGD",
    "Adam",
    "OptimizerType",
    "OptimizerConfig",
    # Training
    "Trainer",
    "SchedulerType",
    "TrainingConfig",
    "SchedulerConfig",
    "RegularizerConfig",
    "EarlyStoppingConfig",
    # Data
    "Dataset",
    "KFoldSplitter",
    # Evaluation
    "evaluate_model",
    "compute_metrics",
]
