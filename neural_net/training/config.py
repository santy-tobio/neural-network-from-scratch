from __future__ import annotations

from dataclasses import dataclass, field

from ..optimizers import OptimizerConfig
from . import SchedulerType


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    type: SchedulerType = SchedulerType.STEP

    # Step scheduler
    step_size: int = 10
    gamma: float = 0.1

    # Exponential scheduler
    decay_rate: float = 0.95

    # Cosine annealing
    T_max: int = 50
    eta_min: float = 0.0

    def __post_init__(self):
        assert self.step_size > 0, "Step size must be positive"
        assert 0 < self.gamma <= 1, "Gamma must be in (0, 1]"
        assert 0 < self.decay_rate <= 1, "Decay rate must be in (0, 1]"
        assert self.T_max > 0, "T_max must be positive"
        assert self.eta_min >= 0, "eta_min must be non-negative"

    def __repr__(self) -> str:
        """Clean representation for logging."""
        if self.type == SchedulerType.STEP:
            return f"SchedulerConfig(type=STEP, step_size={self.step_size}, gamma={self.gamma})"
        return f"SchedulerConfig(type={self.type.name})"


@dataclass
class RegularizerConfig:
    """Configuration for weight regularization (L1/L2)."""

    use_l2: bool = False
    l2_lambda: float = 0.01

    use_l1: bool = False
    l1_lambda: float = 0.01

    def __post_init__(self):
        if self.use_l2:
            assert self.l2_lambda > 0, "L2 lambda must be positive"
        if self.use_l1:
            assert self.l1_lambda > 0, "L1 lambda must be positive"

    def __repr__(self) -> str:
        """Clean representation for logging."""
        active = []
        if self.use_l2:
            active.append(f"L2(λ={self.l2_lambda})")
        if self.use_l1:
            active.append(f"L1(λ={self.l1_lambda})")

        if not active:
            return "RegularizerConfig(None)"
        return f"RegularizerConfig({', '.join(active)})"


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    enabled: bool = False
    patience: int = 10
    monitor: str = "val_loss"  # 'val_loss' or 'val_accuracy'
    min_delta: float = 0.0001
    restore_best_weights: bool = True

    def __post_init__(self):
        assert self.patience > 0, "Patience must be positive"
        assert self.monitor in [
            "val_loss",
            "val_accuracy",
        ], "Monitor must be 'val_loss' or 'val_accuracy'"

    def __repr__(self) -> str:
        """Clean representation for logging."""
        if not self.enabled:
            return "EarlyStoppingConfig(disabled)"
        return (
            f"EarlyStoppingConfig(enabled, patience={self.patience}, "
            f"monitor={self.monitor})"
        )


@dataclass
class TrainingConfig:
    """
    Complete training configuration.

    Combines optimizer, scheduler, regularization, and training hyperparameters.
    """

    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 50
    random_seed: int = 42
    verbose: int = 1  # 0: silent, 1: progress bar, 2: detailed

    # Component configurations (imported from other modules)
    optimizer: OptimizerConfig = field(default_factory=lambda: None)  # type: ignore
    scheduler: SchedulerConfig | None = None
    regularizer: RegularizerConfig = field(default_factory=RegularizerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    # Cross-validation
    use_cross_validation: bool = False
    cv_folds: int = 5
    cv_stratified: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Create default optimizer if not provided
        if self.optimizer is None:
            from neural_net.optimizers import OptimizerConfig

            self.optimizer = OptimizerConfig()

        assert self.batch_size > 0, "Batch size must be positive"
        assert self.epochs > 0, "Epochs must be positive"
        assert self.verbose in [0, 1, 2], "Verbose must be 0, 1, or 2"

        if self.use_cross_validation:
            assert self.cv_folds > 1, "CV folds must be > 1"

    def __repr__(self) -> str:
        """Clean representation for logging."""
        return (
            f"TrainingConfig(\n"
            f"  batch_size={self.batch_size},\n"
            f"  epochs={self.epochs},\n"
            f"  optimizer={self.optimizer},\n"
            f"  scheduler={self.scheduler},\n"
            f"  regularizer={self.regularizer},\n"
            f"  early_stopping={self.early_stopping}\n"
            f")"
        )
