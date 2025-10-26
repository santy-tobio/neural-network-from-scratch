import cupy as cp
from .base_layer import Layer


class Relu(Layer):
    """ReLU activation function"""

    def forward(self, x: cp.ndarray):
        self.mask = cp.array(x > 0, dtype=cp.float32)
        return x * self.mask

    def evaluate(self, x: cp.ndarray) -> cp.ndarray:
        return cp.maximum(0, x)

    def backward(self, prev_grad: cp.ndarray):
        return prev_grad * self.mask
