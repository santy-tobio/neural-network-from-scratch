from neural_net.models import ModelConfig
from neural_net.optimizers import OptimizerConfig, OptimizerType
from neural_net.training import (
    EarlyStoppingConfig,
    RegularizerConfig,
    SchedulerConfig,
    SchedulerType,
    TrainingConfig,
)

from .base import ExperimentConfig

# ============================================================================
# M0: Baseline
# ============================================================================

M0_CONFIG = ExperimentConfig(
    name="M0",
    description="Baseline: 2 hidden layers [128, 64], SGD, no regularization",
    model=ModelConfig(hidden_layers=[128, 64]),
    training=TrainingConfig(
        batch_size=512,
        epochs=100,
        optimizer=OptimizerConfig(
            type=OptimizerType.SGD, learning_rate=0.001, momentum=0.0
        ),
        regularizer=RegularizerConfig(use_l2=False),
        early_stopping=EarlyStoppingConfig(enabled=False),
    ),
)

# ============================================================================
# M1 Variants: Exploring optimizations
# ============================================================================

# ============================================================================
# GROUP 1: Optimizer Variants
# Goal: Compare SGD with momentum vs Adam optimizer
# ============================================================================

M1a_CONFIG = ExperimentConfig(
    name="M1a",
    description="SDG with momentum (0.9) and early stopping, patiece = 5",
    model=ModelConfig(hidden_layers=[128, 64]),
    training=TrainingConfig(
        batch_size=512,
        epochs=50,
        optimizer=OptimizerConfig(
            type=OptimizerType.SGD, learning_rate=0.001, momentum=0.9
        ),
        regularizer=RegularizerConfig(use_l2=False),
        early_stopping=EarlyStoppingConfig(enabled=True, patience=5),
    ),
)

M1b_CONFIG = ExperimentConfig(
    name="M1b",
    description="Adam optimizer (adaptive learning rates) and early stopping, patience = 5",
    model=ModelConfig(hidden_layers=[128, 64]),
    training=TrainingConfig(
        batch_size=512,
        epochs=50,
        optimizer=OptimizerConfig(
            type=OptimizerType.ADAM, learning_rate=0.001, beta1=0.9, beta2=0.999
        ),
        regularizer=RegularizerConfig(use_l2=False),
        early_stopping=EarlyStoppingConfig(enabled=True, patience=5),
    ),
)

# Acá va plot de compración entre M1_a y M1_b para ver cuanto tardan en converger
# cada uno, ponemos early stop, porque sino vamos a estar contando overfit.

# guardamos también métrica en test

# ============================================================================
# GROUP 2: Regularization Techniques
# Goal: Test L2 regularization and early stopping to prevent overfitting
# ============================================================================

M1c_CONFIG = ExperimentConfig(
    name="M1c",
    description="Adam + L2 regularization (λ=0.01)",
    model=ModelConfig(hidden_layers=[128, 64]),
    training=TrainingConfig(
        batch_size=512,
        epochs=100,
        optimizer=OptimizerConfig(
            type=OptimizerType.ADAM, learning_rate=0.001, beta1=0.9, beta2=0.999
        ),
        regularizer=RegularizerConfig(use_l2=True, l2_lambda=0.01),
        early_stopping=EarlyStoppingConfig(enabled=False),
    ),
)

M1d_CONFIG = ExperimentConfig(
    name="M1d",
    description="Adam + L2 + Early Stopping (patience=10)",
    model=ModelConfig(hidden_layers=[128, 64]),
    training=TrainingConfig(
        batch_size=512,
        epochs=100,
        optimizer=OptimizerConfig(
            type=OptimizerType.SGD, learning_rate=0.001, momentum=0.9
        ),
        regularizer=RegularizerConfig(use_l2=True, l2_lambda=0.01),
        early_stopping=EarlyStoppingConfig(
            enabled=True, patience=10, monitor="val_loss"
        ),
    ),
)

# Acá solo vemos el ratio test/val para ver cual overfitea menos
# guardamos también métrica en test

# ============================================================================
# GROUP 3: Learning Rate Scheduling
# Goal: Test different LR decay strategies for better convergence
# ============================================================================

M1e_CONFIG = ExperimentConfig(
    name="M1e",
    description="adam + Linear LR decay",
    model=ModelConfig(hidden_layers=[128, 64]),
    training=TrainingConfig(
        batch_size=512,
        epochs=100,
        optimizer=OptimizerConfig(
            type=OptimizerType.ADAM, learning_rate=0.001, beta1=0.9, beta2=0.999
        ),
        scheduler=SchedulerConfig(
            type=SchedulerType.LINEAR,
            decay_rate=0.00001,  # Decay to ~0.0005 at epoch 50
        ),
        regularizer=RegularizerConfig(use_l2=False),
        early_stopping=EarlyStoppingConfig(enabled=True, patience=5),
    ),
)

M1f_CONFIG = ExperimentConfig(
    name="M1f",
    description="Adam + Exponential LR decay (γ=0.98)",
    model=ModelConfig(hidden_layers=[128, 64]),
    training=TrainingConfig(
        batch_size=512,
        epochs=100,
        optimizer=OptimizerConfig(
            type=OptimizerType.ADAM, learning_rate=0.001, beta1=0.9, beta2=0.999
        ),
        scheduler=SchedulerConfig(
            type=SchedulerType.EXPONENTIAL,
            decay_rate=0.98,  # 0.001 * 0.98^50 ≈ 0.00036
        ),
        regularizer=RegularizerConfig(use_l2=False),
        early_stopping=EarlyStoppingConfig(enabled=True, patience=5),
    ),
)

# Acá va plots de los LR en cada época para ver como decayearon
# vemos después cual dio mejor en test

# ============================================================================
# GROUP 4: Architecture Variants
# Goal: Test different network depths and widths
# ============================================================================

M1g_CONFIG = ExperimentConfig(
    name="M1g",
    description="Deeper architecture [400, 240, 120] + Adam",
    model=ModelConfig(hidden_layers=[400, 240, 120]),
    training=TrainingConfig(
        batch_size=512,
        epochs=50,
        optimizer=OptimizerConfig(
            type=OptimizerType.ADAM, learning_rate=0.001, beta1=0.9, beta2=0.999
        ),
        regularizer=RegularizerConfig(use_l2=False),
        early_stopping=EarlyStoppingConfig(enabled=True, patience=5),
    ),
)

M1h_CONFIG = ExperimentConfig(
    name="M1h",
    description="Wider architecture [370 , 370] + Adam",
    model=ModelConfig(hidden_layers=[370, 370]),
    training=TrainingConfig(
        batch_size=512,
        epochs=50,
        optimizer=OptimizerConfig(
            type=OptimizerType.ADAM, learning_rate=0.001, beta1=0.9, beta2=0.999
        ),
        regularizer=RegularizerConfig(use_l2=False),
        early_stopping=EarlyStoppingConfig(enabled=True, patience=5),
    ),
)

# Acá vemos como afecta el mas capas o mas neuronas por capa en la capacidad
# de generalización del modelo. Solo vemos métricas en test.

# M1g [400, 240, 120]: 444,847 parámetros
# M1h [370, 370]: 445,157 parámetros
# Diferencia: 310 parámetros (0.07%) → prácticamente idénticos.

M1_CORE_CONFIGS = [
    M1a_CONFIG,
    M1b_CONFIG,
    M1c_CONFIG,
    M1d_CONFIG,
    M1e_CONFIG,
    M1f_CONFIG,
    M1g_CONFIG,
    M1h_CONFIG,
]
