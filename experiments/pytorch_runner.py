"""
PyTorch experiment runner.

Provides simple interface for training and evaluating PyTorch models,
compatible with the existing experiment framework.
"""

from typing import Dict, Optional

import cupy as cp
import torch

from neural_net.models.mlp_pytorch import MLPPyTorch
from neural_net.training.trainer_pytorch import TrainerPyTorch

from .configs.pytorch_configs import PyTorchModelConfig


class PyTorchExperiment:
    """
    Wrapper for PyTorch experiments.

    Simplifies training and evaluation of PyTorch models while maintaining
    compatibility with the existing dataset and evaluation infrastructure.
    """

    def __init__(self, config: PyTorchModelConfig):
        """
        Initialize experiment from configuration.

        Args:
            config: PyTorchModelConfig with model and training settings
        """
        self.config = config
        self.model: Optional[MLPPyTorch] = None
        self.trainer: Optional[TrainerPyTorch] = None
        self.history: Optional[Dict] = None

    def build(self):
        """Build model and trainer from configuration."""
        self.model = MLPPyTorch(
            input_dim=self.config.input_dim,
            hidden_layers=self.config.hidden_layers,
            output_dim=self.config.output_dim,
            activation=self.config.activation,
            dropout_rate=self.config.dropout_rate,
            use_batch_norm=self.config.use_batch_norm,
        )

        self.trainer = TrainerPyTorch(
            model=self.model,
            optimizer_type=self.config.optimizer_type,
            learning_rate=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            scheduler_type=self.config.scheduler_type,
            scheduler_params=self.config.scheduler_params,
            early_stopping=self.config.early_stopping,
            patience=self.config.patience,
        )

        print(f"\n{'='*70}")
        print(f"PyTorch Model: {self.config.name}")
        print(f"Description: {self.config.description}")
        print(f"{'='*70}")
        print(f"Architecture:")
        print(f"  Input: {self.config.input_dim}")
        for i, size in enumerate(self.config.hidden_layers, 1):
            print(f"  Hidden {i}: {size} ({self.config.activation})")
            if self.config.dropout_rate > 0:
                print(f"    → Dropout({self.config.dropout_rate})")
            if self.config.use_batch_norm:
                print(f"    → BatchNorm")
        print(f"  Output: {self.config.output_dim}")
        print(f"\nOptimizer: {self.config.optimizer_type.upper()}")
        print(f"  LR: {self.config.learning_rate}")
        if self.config.weight_decay > 0:
            print(f"  Weight decay (L2): {self.config.weight_decay}")
        if self.config.scheduler_type:
            print(f"  Scheduler: {self.config.scheduler_type}")
        print(f"\nTraining:")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Epochs: {self.config.epochs}")
        if self.config.early_stopping:
            print(f"  Early stopping: patience={self.config.patience}")
        print(f"\nTotal parameters: {self.model.count_parameters():,}")
        print(f"{'='*70}\n")

    def train(
        self,
        X_train: cp.ndarray,
        y_train: cp.ndarray,
        X_val: Optional[cp.ndarray] = None,
        y_val: Optional[cp.ndarray] = None,
    ) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features (N, input_dim)
            y_train: Training labels (N,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history dictionary
        """
        if self.trainer is None:
            raise RuntimeError("Must call build() before train()")

        self.history = self.trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=True,
        )

        return self.history

    def get_model_for_evaluation(self):
        """
        Get model wrapper compatible with existing evaluator.

        Returns a wrapper that provides predict() method compatible with CuPy arrays.
        """
        if self.model is None:
            raise RuntimeError("Must call build() before evaluation")

        return PyTorchModelWrapper(self.model, self.trainer.device)


class PyTorchModelWrapper:
    """
    Wrapper to make PyTorch models compatible with CuPy-based evaluator.

    The existing evaluator expects predict() to accept CuPy arrays,
    so we convert CuPy → Torch → CuPy.
    """

    def __init__(self, model: MLPPyTorch, device: str):
        self.model = model
        self.device = device
        self.model.eval()

    def predict(self, X: cp.ndarray) -> cp.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features as CuPy array, either:
               - (N, input_dim) - standard PyTorch format
               - (input_dim, N) - transposed format (for compatibility with custom models)

        Returns:
            Predicted class probabilities as CuPy array (output_dim, N)
        """
        if X.shape[0] == 784:
            X = X.T

        X_np = cp.asnumpy(X)
        X_torch = torch.from_numpy(X_np).to(self.device)

        with torch.no_grad():
            logits = self.model(X_torch)
            probabilities = torch.nn.functional.softmax(logits, dim=1)

        probabilities_np = probabilities.cpu().numpy()
        probabilities_cp = cp.array(probabilities_np)

        return probabilities_cp.T

    def forward(self, X: cp.ndarray) -> cp.ndarray:
        """
        Get raw logits/outputs.

        Args:
            X: Input features as CuPy array, either:
               - (N, input_dim) - standard PyTorch format
               - (input_dim, N) - transposed format (for compatibility with custom models)

        Returns:
            Model outputs as CuPy array (output_dim, N)
        """
        if X.shape[0] == 784:
            X = X.T

        X_np = cp.asnumpy(X)
        X_torch = torch.from_numpy(X_np).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_torch)

        outputs_np = outputs.cpu().numpy()
        outputs_cp = cp.array(outputs_np)

        return outputs_cp.T
