import cupy as cp
from .base_layer import Layer


class Dropout(Layer):
    """Dropout layer for regularization"""

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: cp.ndarray):
        # Randomly drop units during training
        self.mask = (cp.random.rand(*x.shape) > self.drop_prob).astype(cp.float32)
        return x * self.mask / (1.0 - self.drop_prob)  # Scale to keep expected value

    def evaluate(self, x: cp.ndarray) -> cp.ndarray:
        return x  # No dropout during evaluation

    def backward(self, prev_grad: cp.ndarray):
        return (
            prev_grad * self.mask / (1.0 - self.drop_prob)
        )  # Scale gradient accordingly
