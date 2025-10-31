from collections.abc import Callable

import cupy as cp
import numpy as np


class Trainer:
    """Generic trainer that handles the training loop."""

    def __init__(
        self,
        model,
        optimizer,
        loss_fn: Callable[[cp.ndarray, cp.ndarray], float],
        metric_fn: Callable[[cp.ndarray, cp.ndarray], float],
        prepare_batch_fn: Callable[
            [cp.ndarray, cp.ndarray], tuple[cp.ndarray, cp.ndarray]
        ],
        lr_scheduler=None,
        regularizer=None,
        early_stopping=None,
    ):
        """Initialize trainer."""
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.prepare_batch_fn = prepare_batch_fn
        self.lr_scheduler = lr_scheduler
        self.regularizer = regularizer
        self.early_stopping = early_stopping

        # Training history
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_metric": [],
            "val_loss": [],
            "val_metric": [],
        }

    def train_epoch(self, X_train, y_train, batch_size):
        """Train for one epoch using mini-batch SGD.

        X_train and y_train should be NumPy arrays (CPU).
        Only batches are moved to GPU during training.
        """
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Shuffle indices in CPU
        sample_indices = np.arange(n_samples)
        np.random.shuffle(sample_indices)

        epoch_loss = 0.0
        all_predictions = []
        all_labels = []

        # Mini-batch loop
        for batch in range(n_batches):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, n_samples)
            batch_indices = sample_indices[batch_start:batch_end]

            # Get batch data from CPU
            batch_data_X_cpu = X_train[batch_indices]
            batch_data_y_cpu = y_train[batch_indices]
            batch_length = batch_end - batch_start

            # Move batch to GPU
            batch_data_X = cp.asarray(batch_data_X_cpu)
            batch_data_y = cp.asarray(batch_data_y_cpu)

            # Prepare batch using provided function
            X_batch, y_batch_target = self.prepare_batch_fn(batch_data_X, batch_data_y)

            # Forward pass
            outputs = self.model.forward(X_batch)

            # Compute loss using provided function
            batch_loss = self.loss_fn(batch_data_y, outputs)
            epoch_loss += batch_loss * batch_length

            # Store predictions for metric computation
            batch_predictions = cp.argmax(outputs, axis=0)
            all_predictions.append(batch_predictions)
            all_labels.append(batch_data_y)

            # Backward pass (using target format from prepare_batch_fn)
            self.model.backward(y_batch_target)

            # Get gradients
            gradients = self.model.get_gradients()

            # Apply regularization if needed
            if self.regularizer:
                gradients = self.regularizer.apply(gradients, self.model)

            # Compute updates with optimizer
            updates = self.optimizer.step(gradients)

            # Update model parameters
            self.model.update_parameters(updates)

        # Compute epoch metrics
        all_predictions = cp.concatenate(all_predictions)
        all_labels = cp.concatenate(all_labels)

        avg_loss = float(epoch_loss / n_samples)
        metric = self.metric_fn(all_labels, all_predictions)

        return avg_loss, metric

    def validate(self, X_val, y_val, batch_size=2048):
        """Validate the model on validation set using batched inference.

        X_val and y_val should be NumPy arrays (CPU).
        Batches are moved to GPU for validation to avoid OOM errors.
        """
        n_samples = X_val.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        # Batch validation loop
        for batch in range(n_batches):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, n_samples)

            # Get batch from CPU
            X_batch_cpu = X_val[batch_start:batch_end]
            y_batch_cpu = y_val[batch_start:batch_end]
            batch_length = batch_end - batch_start

            # Move batch to GPU
            X_batch_gpu = cp.asarray(X_batch_cpu)
            y_batch_gpu = cp.asarray(y_batch_cpu)

            # Prepare batch
            X_batch, _ = self.prepare_batch_fn(X_batch_gpu, y_batch_gpu)

            # Forward pass
            outputs = self.model.forward(X_batch)

            # Compute batch loss
            batch_loss = self.loss_fn(y_batch_gpu, outputs)
            total_loss += batch_loss * batch_length

            # Store predictions for metric computation
            batch_predictions = cp.argmax(outputs, axis=0)
            all_predictions.append(batch_predictions)
            all_labels.append(y_batch_gpu)

        # Compute overall metrics
        all_predictions = cp.concatenate(all_predictions)
        all_labels = cp.concatenate(all_labels)

        avg_loss = float(total_loss / n_samples)
        metric = self.metric_fn(all_labels, all_predictions)

        return avg_loss, metric

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs: int,
        batch_size: int,
        verbose: bool = True,
    ):
        """Full training loop."""
        for epoch in range(epochs):
            # Train one epoch
            train_loss, train_metric = self.train_epoch(X_train, y_train, batch_size)

            # Validate
            val_loss, val_metric = self.validate(X_val, y_val)

            # Update learning rate if scheduler is provided
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch)

            # Check early stopping
            if self.early_stopping and verbose:
                if self.early_stopping(val_loss):
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["train_metric"].append(train_metric)
            self.history["val_loss"].append(val_loss)
            self.history["val_metric"].append(val_metric)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {train_loss:.4f}, Metric: {train_metric:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}"
                )

        return self.history
