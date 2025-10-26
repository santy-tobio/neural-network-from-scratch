from abc import ABC, abstractmethod


class LRScheduler(ABC):
    """Base class for learning rate schedulers"""

    def __init__(self, optimizer, initial_lr: float):
        self.optimizer = optimizer
        self.initial_lr = initial_lr

    @abstractmethod
    def step(self, epoch: int):
        """Update learning rate based on epoch"""


class LinearScheduler(LRScheduler):
    """
    Linear learning rate decay with saturation.

    lr = max(min_lr, initial_lr - decay_rate * epoch)
    """

    def __init__(
        self, optimizer, initial_lr: float, decay_rate: float, min_lr: float = 1e-6
    ):
        super().__init__(optimizer, initial_lr)
        self.decay_rate = decay_rate
        self.min_lr = min_lr

    def step(self, epoch: int):
        """Update learning rate linearly"""
        new_lr = max(self.min_lr, self.initial_lr - self.decay_rate * epoch)
        self.optimizer.learning_rate = new_lr


class ExponentialScheduler(LRScheduler):
    """
    Exponential learning rate decay.

    lr = initial_lr * gamma^epoch
    """

    def __init__(self, optimizer, initial_lr: float, gamma: float = 0.95):
        super().__init__(optimizer, initial_lr)
        self.gamma = gamma

    def step(self, epoch: int):
        """Update learning rate exponentially"""
        new_lr = self.initial_lr * (self.gamma**epoch)
        self.optimizer.learning_rate = new_lr
