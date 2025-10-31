"""
Experiment runner - builds training components from configuration.
"""

import cupy as cp

from neural_net import create_training_components
from neural_net.evaluation.metrics import accuracy, cross_entropy
from neural_net.utils.training_helpers import prepare_classification_batch

from .configs.base import ExperimentConfig


class Experiment:
    """
    Creates model, optimizer, and trainer from an ExperimentConfig.

    Follows the pattern used in Jupyter notebook (jupyter_reference.py).
    """

    def __init__(self, config: ExperimentConfig, dataset):
        """
        Initialize experiment.

        Args:
            config: ExperimentConfig with model and training settings
            dataset: Dataset object with train/val/test splits
        """
        self.config = config
        self.dataset = dataset

        cp.random.seed(config.training.random_seed)

        self.components: dict | None = None
        self.model = None
        self.trainer = None

    def build(self):
        """
        Build model, optimizer, and trainer from configuration.

        After calling this, you can access:
        - self.model: MLP model
        - self.trainer: Trainer with optimizer, scheduler, etc.
        - self.components: dict with all components
        """
        input_dim = self.dataset.X_train.shape[1]
        output_dim = len(cp.unique(self.dataset.y_train))

        def prepare_batch(X_raw, y_raw):
            return prepare_classification_batch(X_raw, y_raw, output_dim)

        self.components = create_training_components(
            model_config=self.config.model,
            training_config=self.config.training,
            input_dim=input_dim,
            output_dim=output_dim,
            loss_fn=cross_entropy,
            metric_fn=accuracy,
            prepare_batch_fn=prepare_batch,
        )

        self.model = self.components["model"]
        self.trainer = self.components["trainer"]
