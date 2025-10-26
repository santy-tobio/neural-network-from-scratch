from .config import EarlyStoppingConfig, RegularizerConfig, SchedulerConfig
from .early_stopping import EarlyStopping
from .lr_schedulers import LRScheduler
from .regularizers import L2Regularizer


def create_scheduler(
    config: SchedulerConfig, optimizer, initial_lr: float
) -> LRScheduler:
    """Create a learning rate scheduler from configuration."""
    scheduler_class = config.type.value
    return scheduler_class.from_config(config, optimizer, initial_lr)


def create_regularizer(config: RegularizerConfig) -> L2Regularizer | None:
    """
    Create a regularizer from configuration.

    Note: Currently only supports L2 regularization.
    """
    if config.use_l2:
        return L2Regularizer(lambda_=config.l2_lambda)
    # Note: L1 regularization not implemented yet
    # Note: Dropout is handled at the model level, not here
    return None


def create_early_stopping(config: EarlyStoppingConfig) -> EarlyStopping | None:
    """Create an early stopping instance from configuration."""
    if not config.enabled:
        return None

    return EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
    )
