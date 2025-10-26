from enum import Enum

from .base import Layer
from .ce_softmax import CESoftmax
from .dropout import Dropout
from .linear import Linear
from .relu import Relu

__all__ = ["Layer", "Linear", "Relu", "CESoftmax", "Dropout", "LayerType"]


class LayerType(Enum):
    """Types of layers available in the neural network."""

    LINEAR = Linear
    RELU = Relu
    SOFTMAX = CESoftmax
    DROPOUT = Dropout
