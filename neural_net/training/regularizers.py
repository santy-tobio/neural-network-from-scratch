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
