from neural_net.optimizers import OptimizerType

from .base_config import (
    EarlyStoppingConfig,
    ExperimentConfig,
    OptimizerConfig,
    RegularizerConfig,
)

M0_CONFIG = ExperimentConfig(
    name="M0",
    description="Baseline: 2 hidden layers [128, 64], SGD, no regularization",
    hidden_layers=[128, 64],
    batch_size=256,
    epochs=50,
    optimizer=OptimizerConfig(
        type=OptimizerType.SGD, learning_rate=0.001, momentum=0.0
    ),
    regularizer=RegularizerConfig(use_l2=False),
    early_stopping=EarlyStoppingConfig(enabled=False),
)

M1_CONFIG = ExperimentConfig(
    name="M1",
    description="With momentum: SGD with momentum=0.9",
    hidden_layers=[128, 64],
    batch_size=256,
    epochs=50,
    optimizer=OptimizerConfig(
        type=OptimizerType.SGD, learning_rate=0.001, momentum=0.9
    ),
    regularizer=RegularizerConfig(use_l2=False),
    early_stopping=EarlyStoppingConfig(enabled=False),
)

M2_CONFIG = ExperimentConfig(
    name="M2",
    description="With L2 regularization: lambda=0.01",
    hidden_layers=[128, 64],
    batch_size=256,
    epochs=50,
    optimizer=OptimizerConfig(
        type=OptimizerType.SGD, learning_rate=0.001, momentum=0.9
    ),
    regularizer=RegularizerConfig(use_l2=True, l2_lambda=0.01),
    early_stopping=EarlyStoppingConfig(enabled=False),
)

M3_CONFIG = ExperimentConfig(
    name="M3",
    description="With early stopping: patience=10",
    hidden_layers=[128, 64],
    batch_size=256,
    epochs=100,
    optimizer=OptimizerConfig(
        type=OptimizerType.SGD, learning_rate=0.001, momentum=0.9
    ),
    regularizer=RegularizerConfig(use_l2=True, l2_lambda=0.01),
    early_stopping=EarlyStoppingConfig(enabled=True, patience=10, monitor="val_loss"),
)

ADAM_CONFIG = ExperimentConfig(
    name="M_ADAM",
    description="Using Adam optimizer",
    hidden_layers=[128, 64],
    batch_size=256,
    epochs=50,
    optimizer=OptimizerConfig(
        type=OptimizerType.ADAM, learning_rate=0.001, beta1=0.9, beta2=0.999
    ),
    regularizer=RegularizerConfig(use_l2=False),
    early_stopping=EarlyStoppingConfig(enabled=False),
)

# Cross-validation example
CV_CONFIG = ExperimentConfig(
    name="M_CV",
    description="5-Fold stratified cross-validation",
    hidden_layers=[128, 64],
    batch_size=256,
    epochs=50,
    optimizer=OptimizerConfig(
        type=OptimizerType.SGD, learning_rate=0.001, momentum=0.9
    ),
    regularizer=RegularizerConfig(use_l2=True, l2_lambda=0.01),
    early_stopping=EarlyStoppingConfig(enabled=False),
    use_cross_validation=True,
    cv_folds=5,
    cv_stratified=True,
)
