from dataclasses import dataclass, field
from typing import Any

from neural_net.layers import LayerType
from neural_net.optimizers import OptimizerType
from neural_net.training import SchedulerType


@dataclass
class OptimizerConfig:
    """Configuration for optimizer"""

    type: OptimizerType = OptimizerType.SGD
    learning_rate: float = 0.001

    # SGD-specific
    momentum: float = 0.0
    nesterov: bool = False

    # Adam-specific
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self):
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert 0 <= self.momentum < 1, "Momentum must be in [0, 1)"
        assert 0 < self.beta1 < 1, "Beta1 must be in (0, 1)"
        assert 0 < self.beta2 < 1, "Beta2 must be in (0, 1)"


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler"""

    type: SchedulerType = SchedulerType.STEP

    # Step scheduler
    step_size: int = 10
    gamma: float = 0.1

    def __post_init__(self):
        assert self.step_size > 0, "Step size must be positive"
        assert 0 < self.gamma < 1, "Gamma must be in (0, 1)"


@dataclass
class RegularizerConfig:
    """Configuration for regularization"""

    use_l2: bool = False
    l2_lambda: float = 0.01

    use_l1: bool = False
    l1_lambda: float = 0.01

    use_dropout: bool = False
    dropout_rate: float = 0.5

    def __post_init__(self):
        if self.use_l2:
            assert self.l2_lambda > 0, "L2 lambda must be positive"
        if self.use_l1:
            assert self.l1_lambda > 0, "L1 lambda must be positive"
        if self.use_dropout:
            assert 0 < self.dropout_rate < 1, "Dropout rate must be in (0, 1)"


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping"""

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


@dataclass
class ExperimentConfig:
    """
    Configuration for a single experiment.

    This allows you to define all hyperparameters in one place
    and ensures reproducibility.
    """

    # Experiment metadata
    name: str  # e.g., 'M0', 'M1', 'M2', 'M3'
    description: str
    hidden_layers: list[int]  # e.g., [128, 64]

    # Optional fields with defaults
    random_seed: int = 42
    activation: LayerType = LayerType.RELU
    output_activation: LayerType = LayerType.SOFTMAX
    batch_size: int = 32
    epochs: int = 50

    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            type=OptimizerType.SGD, learning_rate=0.001
        )
    )  # Required field with default

    scheduler: SchedulerConfig | None = None
    regularizer: RegularizerConfig = field(default_factory=RegularizerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    # ===== Other Settings =====
    verbose: int = 1  # 0: silent, 1: progress bar, 2: detailed

    # ===== Cross-validation =====
    use_cross_validation: bool = False
    cv_folds: int = 5
    cv_stratified: bool = True

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    def __post_init__(self):
        """Validate configuration after initialization"""
        # Architecture validation
        assert len(self.hidden_layers) > 0, "Must have at least one hidden layer"
        assert all(
            units > 0 for units in self.hidden_layers
        ), "All layer sizes must be positive"

        # Training validation
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.epochs > 0, "Epochs must be positive"

        # CV validation
        if self.use_cross_validation:
            assert self.cv_folds > 1, "CV folds must be > 1"

    def __repr__(self) -> str:
        """Clean representation for logging"""
        return (
            f"ExperimentConfig(\n"
            f"  name='{self.name}',\n"
            f"  description='{self.description}',\n"
            f"  architecture={self.hidden_layers},\n"
            f"  batch_size={self.batch_size},\n"
            f"  epochs={self.epochs},\n"
            f"  optimizer={self.optimizer.type.value} (lr={self.optimizer.learning_rate}),\n"
            f"  scheduler={self.scheduler.type.value if self.scheduler else None},\n"
            f"  regularization={'L2' if self.regularizer.use_l2 else 'None'},\n"
            f"  early_stopping={self.early_stopping.enabled}\n"
            f")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "optimizer": {
                "type": self.optimizer.type.value,
                "learning_rate": self.optimizer.learning_rate,
                "momentum": self.optimizer.momentum,
                "beta1": self.optimizer.beta1,
                "beta2": self.optimizer.beta2,
            },
            "scheduler": (
                {
                    "type": self.scheduler.type.value if self.scheduler else None,
                }
                if self.scheduler
                else None
            ),
            "regularizer": {
                "use_l2": self.regularizer.use_l2,
                "l2_lambda": self.regularizer.l2_lambda,
            },
            "early_stopping": {
                "enabled": self.early_stopping.enabled,
                "patience": self.early_stopping.patience,
            },
            "random_seed": self.random_seed,
        }
