import cupy as cp

from .base import Layer


class Linear(Layer):
    """Fully connected linear layer"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        self.weights = cp.random.randn(output_dim, input_dim).astype(
            cp.float32
        ) * cp.sqrt(2.0 / input_dim)
        self.bias = cp.zeros((output_dim, 1), dtype=cp.float32)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        # weight: (output_dim, input_dim) @ x: (input_dim, batch_size) -> (output_dim, batch_size)
        # bias: (output_dim, batch_size)
        # return: (output_dim, batch_size)
        self.input = x
        return self.weights @ x + self.bias

    def evaluate(self, x: cp.ndarray) -> cp.ndarray:
        return self.weights @ x + self.bias

    def backward(self, prev_grad: cp.ndarray) -> cp.ndarray:
        # prev_grad: (output_dim, batch_size), input: (input_dim, batch_size)
        # Weights: DL/DZ3 * (a^i-1)^T
        self.grad_weights = prev_grad @ self.input.T  # (output_dim, input_dim)
        # Bias gradient is the sum over the batch dimension -> (output_dim, 1)
        self.grad_bias = cp.sum(prev_grad, axis=1, keepdims=True)

        # Layer is independant of the batch size (which is handled in the optimizer)

        return self.weights.T @ prev_grad
