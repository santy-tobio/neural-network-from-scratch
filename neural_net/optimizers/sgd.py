import cupy as cp
from .base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer with momentum and mini-batch SGD.
    """

    def __init__(self, learning_rate: float, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None  # Will store momentum buffers

    def step(
        self, gradients: list[tuple[cp.ndarray, cp.ndarray]]
    ) -> list[tuple[cp.ndarray, cp.ndarray]]:
        """
        Compute parameter updates using gradient descent.
        """
        if self.momentum > 0 and self.velocity is None:
            self.velocity = []
            for grad_w, grad_b in gradients:
                self.velocity.append((cp.zeros_like(grad_w), cp.zeros_like(grad_b)))

        updates = []

        if self.momentum > 0:
            for i, (grad_w, grad_b) in enumerate(gradients):
                v_w, v_b = self.velocity[i]

                v_w = self.momentum * v_w + self.learning_rate * grad_w
                v_b = self.momentum * v_b + self.learning_rate * grad_b
                self.velocity[i] = (v_w, v_b)
                updates.append((v_w, v_b))
        else:  # Standard SGD
            for grad_w, grad_b in gradients:
                update_w = self.learning_rate * grad_w
                update_b = self.learning_rate * grad_b
                updates.append((update_w, update_b))

        return updates

    def reset(self):
        """Reset momentum buffers"""
        self.velocity = None
