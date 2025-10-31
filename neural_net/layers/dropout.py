import cupy as cp

from .base import Layer


class Dropout(Layer):
    """Dropout layer for regularization"""

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: cp.ndarray):
        self.mask = (cp.random.rand(*x.shape) > self.drop_prob).astype(cp.float32)
        return x * self.mask / (1.0 - self.drop_prob)

    def evaluate(self, x: cp.ndarray) -> cp.ndarray:
        return x

    def backward(self, prev_grad: cp.ndarray):
        return prev_grad * self.mask / (1.0 - self.drop_prob)
