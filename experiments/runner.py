"""
Experiment runner.
Ties together model, optimizer, trainer, and evaluator.
"""

import cupy as cp

from neural_net.evaluation import Evaluator
from neural_net.models import MLP
from neural_net.optimizers import SGD, Adam
from neural_net.training import (
    EarlyStopping,
    ExponentialScheduler,
    L2Regularizer,
    LinearScheduler,
    Trainer,
)

from .configs.base_config import ExperimentConfig


class Experiment:
    """
    Runs a complete experiment given a configuration.

    This is the high-level interface that puts everything together.
    """

    def __init__(self, config: ExperimentConfig, dataset):
        self.config = config
        self.dataset = dataset

        # Set random seed
        cp.random.seed(config.random_seed)

        # Initialize components
        self.model = None
        self.optimizer = None
        self.trainer = None
        self.evaluator = Evaluator()

    def build_model(self):
        """Build model based on config"""
        input_dim = self.dataset.X_devel.shape[1]  # Already flattened
        output_dim = len(cp.unique(self.dataset.y_devel))

        self.model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=self.config.hidden_layers,
            batch_size=self.config.batch_size,
        )

    def build_optimizer(self):
        """Build optimizer based on config"""
        from neural_net.optimizers import OptimizerType

        if self.config.optimizer.type == OptimizerType.SGD:
            self.optimizer = SGD(
                learning_rate=self.config.optimizer.learning_rate,
                momentum=self.config.optimizer.momentum,
            )
        elif self.config.optimizer.type == OptimizerType.ADAM:
            self.optimizer = Adam(
                learning_rate=self.config.optimizer.learning_rate,
                beta1=self.config.optimizer.beta1,
                beta2=self.config.optimizer.beta2,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer.type}")

    def build_trainer(self):
        """Build trainer with all components"""
        # Learning rate scheduler
        lr_scheduler = None
        if self.config.scheduler:
            from neural_net.training.lr_schedulers import SchedulerType

            if self.config.scheduler.type == SchedulerType.LINEAR:
                lr_scheduler = LinearScheduler(
                    self.optimizer,
                    self.config.optimizer.learning_rate,
                    self.config.scheduler.decay_rate,
                )
            elif self.config.scheduler.type == SchedulerType.EXPONENTIAL:
                lr_scheduler = ExponentialScheduler(
                    self.optimizer,
                    self.config.optimizer.learning_rate,
                    self.config.scheduler.gamma,
                )

        # Regularizer
        regularizer = None
        if self.config.regularizer.use_l2:
            regularizer = L2Regularizer(self.config.regularizer.l2_lambda)

        # Early stopping
        early_stopping = None
        if self.config.early_stopping.enabled:
            early_stopping = EarlyStopping(patience=self.config.early_stopping.patience)

        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=lr_scheduler,
            regularizer=regularizer,
            early_stopping=early_stopping,
        )

    def run(self):
        """Run the complete experiment"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {self.config.name}")
        print(f"Description: {self.config.description}")
        print(f"{'='*60}\n")

        # Build components
        self.build_model()
        self.build_optimizer()
        self.build_trainer()

        # Get data (updated for dev/test split)
        X_train, y_train = self.dataset.get_devel()
        X_test, y_test = self.dataset.get_test()

        # Use a portion of devel as validation
        val_split = 0.15
        n_val = int(len(X_train) * val_split)
        X_val, y_val = X_train[-n_val:], y_train[-n_val:]
        X_train, y_train = X_train[:-n_val], y_train[:-n_val]

        # Train
        print("Training...")
        history = self.trainer.train(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=True,
        )

        # Evaluate on test
        print("\nEvaluating on test set...")
        test_metrics = self.evaluator.evaluate(
            self.model, X_test, y_test, self.config.name
        )

        print("\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {test_metrics['f1_macro']:.4f}")

        return {"history": history, "test_metrics": test_metrics, "model": self.model}
