import cupy as cp

from neural_net import create_training_components
from neural_net.evaluation import evaluate_model
from neural_net.evaluation.metrics import accuracy, cross_entropy
from neural_net.utils.training_helpers import prepare_classification_batch

from .configs.base import ExperimentConfig


class Experiment:
    """
    Runs a complete experiment given a configuration.
    """

    def __init__(self, config: ExperimentConfig, dataset):
        self.config = config
        self.dataset = dataset

        # Set random seed
        cp.random.seed(config.training.random_seed)

        # Components will be created in build()
        self.components: dict | None = None
        self.model = None
        self.trainer = None

    def build(self):
        """Build all components from configuration using factory functions."""
        # Get dataset dimensions
        input_dim = self.dataset.X_devel.shape[1]
        output_dim = len(cp.unique(self.dataset.y_devel))

        def prepare_batch(X_raw, y_raw):
            return prepare_classification_batch(X_raw, y_raw, output_dim)

        # Create all components using the factory
        self.components = create_training_components(
            model_config=self.config.model,
            training_config=self.config.training,
            input_dim=input_dim,
            output_dim=output_dim,
            loss_fn=cross_entropy,
            metric_fn=accuracy,
            prepare_batch_fn=prepare_batch,
        )

        # Extract commonly used components
        self.model = self.components["model"]
        self.trainer = self.components["trainer"]

    def run(self):
        """Run the complete experiment"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {self.config.name}")
        print(f"Description: {self.config.description}")
        print(f"{'='*60}\n")

        # Build all components
        self.build()

        # Ensure components are built
        assert self.components is not None
        assert self.model is not None
        assert self.trainer is not None

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
            epochs=self.config.training.epochs,
            batch_size=self.config.training.batch_size,
            verbose=True,
        )

        # Evaluate on test
        print("\nEvaluating on test set...")
        test_metrics = evaluate_model(self.model, X_test, y_test)

        print("\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {test_metrics['f1_macro']:.4f}")

        return {
            "history": history,
            "test_metrics": test_metrics,
            "model": self.model,
            "components": self.components,
        }
