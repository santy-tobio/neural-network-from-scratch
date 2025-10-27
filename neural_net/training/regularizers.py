import cupy as cp


class L2Regularizer:
    """L2 (Ridge) regularization for weight decay."""

    def __init__(self, lambda_: float = 0.01):
        self.lambda_ = lambda_

    def __call__(self, weights: cp.ndarray) -> float:
        """Compute L2 regularization penalty."""
        return 0.5 * self.lambda_ * cp.sum(weights**2)

    def gradient(self, weights: cp.ndarray) -> cp.ndarray:
        """Compute gradient of L2 regularization."""
        return self.lambda_ * weights

    def apply(
        self, gradients: list[tuple[cp.ndarray, cp.ndarray]], model
    ) -> list[tuple[cp.ndarray, cp.ndarray]]:
        """Apply L2 regularization to gradients for each Linear layer in the model."""
        linear_idx = 0
        for layer in getattr(model, "layers", []):
            if hasattr(layer, "weights"):
                grad_weights, grad_bias = gradients[linear_idx]
                grad_weights += self.gradient(layer.weights)
                gradients[linear_idx] = (grad_weights, grad_bias)
                linear_idx += 1
        return gradients
