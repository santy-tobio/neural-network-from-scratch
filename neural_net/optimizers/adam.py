import cupy as cp

from .base import BaseOptimizer


class Adam(BaseOptimizer):
    """Adam optimizer (Adaptive Moment Estimation)."""

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: list[tuple[cp.ndarray, cp.ndarray]] | None = None
        self.v: list[tuple[cp.ndarray, cp.ndarray]] | None = None
        self.t = 0

    @classmethod
    def from_config(cls, config):
        """Create Adam optimizer from configuration."""
        return cls(
            learning_rate=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
        )

    def step(
        self, gradients: list[tuple[cp.ndarray, cp.ndarray]]
    ) -> list[tuple[cp.ndarray, cp.ndarray]]:
        """Compute parameter updates using Adam algorithm."""
        self.t += 1

        if self.m is None:
            self.m = [
                (cp.zeros_like(grad_w), cp.zeros_like(grad_b))
                for grad_w, grad_b in gradients
            ]
            self.v = [
                (cp.zeros_like(grad_w), cp.zeros_like(grad_b))
                for grad_w, grad_b in gradients
            ]

        updates = []

        for i, (grad_w, grad_b) in enumerate(gradients):

            m_w = self.beta1 * self.m[i][0] + (1 - self.beta1) * grad_w
            m_b = self.beta1 * self.m[i][1] + (1 - self.beta1) * grad_b

            v_w = self.beta2 * self.v[i][0] + (1 - self.beta2) * (grad_w**2)
            v_b = self.beta2 * self.v[i][1] + (1 - self.beta2) * (grad_b**2)

            self.m[i] = (m_w, m_b)
            self.v[i] = (v_w, v_b)

            m_w_hat = m_w / (1 - self.beta1**self.t)
            m_b_hat = m_b / (1 - self.beta1**self.t)

            v_w_hat = v_w / (1 - self.beta2**self.t)
            v_b_hat = v_b / (1 - self.beta2**self.t)

            update_w = self.learning_rate * m_w_hat / (cp.sqrt(v_w_hat) + self.epsilon)
            update_b = self.learning_rate * m_b_hat / (cp.sqrt(v_b_hat) + self.epsilon)

            updates.append((update_w, update_b))

        return updates

    def reset(self):
        """Reset Adam state (moving averages and timestep)."""
        self.m = None
        self.v = None
        self.t = 0
