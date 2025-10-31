"""
PyTorch-based Trainer for neural network training.

This module provides a trainer compatible with PyTorch models,
designed to mirror the interface of the custom CuPy/NumPy trainer.
"""

import time
from typing import Dict, Optional, Tuple

import cupy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TrainerPyTorch:
    """
    Trainer for PyTorch models.

    Provides training loop with support for:
    - Adam and SGD optimizers
    - Learning rate scheduling
    - Early stopping
    - L2 regularization (weight decay)
    - GPU acceleration
    - Progress tracking

    Args:
        model: PyTorch nn.Module to train
        optimizer_type: 'adam' or 'sgd'
        learning_rate: Initial learning rate
        momentum: Momentum for SGD (ignored for Adam)
        weight_decay: L2 regularization strength
        scheduler_type: 'linear', 'exponential', or None
        scheduler_params: Dict with scheduler-specific parameters
        early_stopping: Whether to use early stopping
        patience: Number of epochs to wait for improvement
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_type: str = "adam",
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler_type: Optional[str] = None,
        scheduler_params: Optional[Dict] = None,
        early_stopping: bool = False,
        patience: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.early_stopping = early_stopping
        self.patience = patience

        # Loss function (CrossEntropyLoss expects logits)
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Learning rate scheduler
        self.scheduler = None
        if scheduler_type:
            scheduler_params = scheduler_params or {}
            if scheduler_type.lower() == "linear":
                # Linear decay
                decay_rate = scheduler_params.get("decay_rate", 0.1)
                total_epochs = scheduler_params.get("total_epochs", 100)
                self.scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=decay_rate,
                    total_iters=total_epochs,
                )
            elif scheduler_type.lower() == "exponential":
                # Exponential decay
                gamma = scheduler_params.get("decay_rate", 0.95)
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=gamma
                )
            else:
                raise ValueError(f"Unknown scheduler: {scheduler_type}")

        # Early stopping state
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def train(
        self,
        X_train: cp.ndarray,
        y_train: cp.ndarray,
        X_val: Optional[cp.ndarray] = None,
        y_val: Optional[cp.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 512,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Train the model.

        Args:
            X_train: Training features (N, input_dim) - CuPy array
            y_train: Training labels (N,) - CuPy array
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print progress

        Returns:
            Dictionary with training history (loss and accuracy curves)
        """
        # Convert CuPy arrays to PyTorch tensors
        X_train_torch = self._cupy_to_torch(X_train)
        y_train_torch = self._cupy_to_torch(y_train).long()

        # Create DataLoader
        train_dataset = TensorDataset(X_train_torch, y_train_torch)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # Prepare validation data if provided
        if X_val is not None and y_val is not None:
            X_val_torch = self._cupy_to_torch(X_val)
            y_val_torch = self._cupy_to_torch(y_val).long()
            val_dataset = TensorDataset(X_val_torch, y_val_torch)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )
        else:
            val_loader = None

        # Training history
        history = {
            "train_loss": [],
            "train_metric": [],
            "val_loss": [],
            "val_metric": [],
            "learning_rates": [],
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f"Training PyTorch model on {self.device}")
            print(f"{'='*70}")

        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()

            # Train one epoch
            train_loss, train_acc = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            history["train_metric"].append(train_acc)

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_metric"].append(val_acc)
            else:
                val_loss, val_acc = None, None

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["learning_rates"].append(current_lr)

            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start
                msg = f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s) - "
                msg += f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
                msg += f" - LR: {current_lr:.6f}"
                print(msg)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Early stopping
            if self.early_stopping and val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.patience:
                    if verbose:
                        print(
                            f"\n⚠ Early stopping triggered at epoch {epoch+1} "
                            f"(patience={self.patience})"
                        )
                    break

        if verbose:
            print(f"{'='*70}\n")

        return history

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                # Track metrics
                total_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def _cupy_to_torch(self, arr: cp.ndarray) -> torch.Tensor:
        """Convert CuPy array to PyTorch tensor."""
        # CuPy → NumPy → PyTorch
        np_arr = cp.asnumpy(arr)
        return torch.from_numpy(np_arr)

    def get_model(self) -> nn.Module:
        """Get the trained model."""
        return self.model
