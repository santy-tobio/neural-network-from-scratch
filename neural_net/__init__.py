"""
Neural Network Library

A modular neural network implementation with MLP models,
optimizers (SGD, Adam), training utilities, and evaluation metrics.
"""

__version__ = "0.1.0"

# Import key classes for convenient access
from .data import Dataset, KFoldSplitter
from .evaluation import Evaluator, compute_metrics
from .layers import CESoftmax, Dropout, Layer, LayerType, Linear, Relu
from .models import MLP, BaseMLP
from .optimizers import SGD, Adam, OptimizerType
from .training import SchedulerType, Trainer

__all__ = [
    # Models
    "MLP",
    "BaseMLP",
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
    # Training
    "Trainer",
    "SchedulerType",
    # Data
    "Dataset",
    "KFoldSplitter",
    # Evaluation
    "Evaluator",
    "compute_metrics",
]
