from collections.abc import Callable

from .models import ModelConfig, create_mlp
from .optimizers import create_optimizer
from .training import (
    Trainer,
    TrainingConfig,
    create_early_stopping,
    create_regularizer,
    create_scheduler,
)


def create_training_components(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    input_dim: int,
    output_dim: int,
    loss_fn: Callable,
    metric_fn: Callable,
    prepare_batch_fn: Callable,
) -> dict:
    """
    Create all training components from configurations.

    This is the main orchestrator that creates:
    - Model (MLP)
    - Optimizer
    - Learning rate scheduler (optional)
    - Regularizer (optional)
    - Early stopping (optional)
    - Trainer (that combines all of the above)
    """
    # Create model
    model = create_mlp(
        model_config=model_config,
        input_dim=input_dim,
        output_dim=output_dim,
    )

    # Create optimizer
    optimizer = create_optimizer(training_config.optimizer)

    # Create scheduler (optional)
    scheduler = None
    if training_config.scheduler is not None:
        scheduler = create_scheduler(
            config=training_config.scheduler,
            optimizer=optimizer,
            initial_lr=training_config.optimizer.learning_rate,
        )

    # Create regularizer (optional)
    regularizer = create_regularizer(training_config.regularizer)

    # Create early stopping (optional)
    early_stopping = create_early_stopping(training_config.early_stopping)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        prepare_batch_fn=prepare_batch_fn,
        lr_scheduler=scheduler,
        regularizer=regularizer,
        early_stopping=early_stopping,
    )

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "regularizer": regularizer,
        "early_stopping": early_stopping,
        "trainer": trainer,
    }
