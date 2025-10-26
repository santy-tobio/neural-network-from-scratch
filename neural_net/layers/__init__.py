from enum import Enum

from .base_layer import Layer
from .dropout import Dropout
from .linear import Linear
from .relu import Relu
from .ce_softmax import CESoftmax

__all__ = ["Layer", "Linear", "Relu", "CESoftmax", "Dropout", "LayerType"]


class LayerType(Enum):
    """Types of layers available in the neural network."""

    LINEAR = "linear"
    RELU = "relu"
    SOFTMAX = "softmax"
    DROPOUT = "dropout"
