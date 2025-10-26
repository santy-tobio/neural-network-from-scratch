from abc import ABC, abstractmethod

import cupy as cp


class Layer(ABC):
    """Base class for all layers"""

    def __init__(self, input_dim: int | None = None, output_dim: int | None = None):
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: cp.ndarray):
        """
        Forward pass through the layer.
        """

    @abstractmethod
    def evaluate(self, x: cp.ndarray):
        """
        Evaluate layer output without storing intermediate values.
        """

    @abstractmethod
    def backward(self, prev_grad: cp.ndarray):
        """
        Backward pass through the layer.
        """
