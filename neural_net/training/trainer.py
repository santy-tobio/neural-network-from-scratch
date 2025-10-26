"""
Generic trainer class that works with any model and optimizer.
This is the KEY abstraction that ensures fair comparison.
"""


import cupy as cp


class Trainer:
    """
    Generic trainer that handles the training loop.

    This class ensures that ALL models (M0, M1, M2, M3) are trained
    using the same protocol, making comparisons fair.

    Args:
        model: Any model implementing BaseMLP interface
        optimizer: Any optimizer implementing BaseOptimizer interface
        lr_scheduler: Optional learning rate scheduler
        regularizer: Optional regularizer (L2, etc.)
        early_stopping: Optional early stopping callback
    """

    def __init__(
        self, model, optimizer, lr_scheduler=None, regularizer=None, early_stopping=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.regularizer = regularizer
        self.early_stopping = early_stopping

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_epoch(self, X_train, y_train, batch_size):
        """
        Train for one epoch using mini-batch SGD.

        Args:
            X_train: Training data (n_samples, 28, 28)
            y_train: Training labels (n_samples,)
            batch_size: Batch size for mini-batch SGD

        Returns:
            Average loss and accuracy for the epoch
        """
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        sample_indices = cp.arange(n_samples)

        # Shuffle data
        cp.random.shuffle(sample_indices)
        X_shuffled = X_train[sample_indices]
        y_shuffled = y_train[sample_indices]

        epoch_correct = 0
        epoch_loss = 0.0

        # Mini-batch loop
        for batch in range(n_batches):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, n_samples)
            batch_data_X = X_shuffled[batch_start:batch_end]
            batch_data_y = y_shuffled[batch_start:batch_end]
            batch_length = batch_end - batch_start

            # Prepare batch: (batch_length, 28, 28) -> (784, batch_length)
            X_batch = batch_data_X.reshape(batch_length, -1).T.astype(cp.float32)

            # One-hot encode labels: (output_dim, batch_length)
            y_batch = cp.zeros((self.model.output_dim, batch_length), dtype=cp.float32)
            y_batch[batch_data_y.astype(int), cp.arange(batch_length)] = 1.0

            # Forward pass
            outputs = self.model.forward(X_batch)

            # Compute cross-entropy loss
            loss = -cp.sum(y_batch * cp.log(outputs + 1e-8)) / batch_length
            epoch_loss += loss * batch_length

            # Compute accuracy
            batch_correct = cp.sum(cp.argmax(outputs, axis=0) == batch_data_y)
            epoch_correct += batch_correct

            # Backward pass
            self.model.backward(y_batch)

            # Get gradients
            gradients = self.model.get_gradients()

            # Apply regularization if needed
            if self.regularizer:
                # TODO: Get model weights for L2 regularization
                pass

            # Compute updates with optimizer
            updates = self.optimizer.step(gradients)

            # Update model parameters
            self.model.update_parameters(updates)

        # Return average loss and accuracy
        avg_loss = float(epoch_loss / n_samples)
        accuracy = float(epoch_correct / n_samples)

        return avg_loss, accuracy

    def validate(self, X_val, y_val):
        """
        Validate the model on validation set.

        Args:
            X_val: Validation data (n_samples, 28, 28)
            y_val: Validation labels (n_samples,)

        Returns:
            Validation loss and accuracy
        """
        n_samples = X_val.shape[0]

        # Prepare data: (n_samples, 28, 28) -> (784, n_samples)
        X_batch = X_val.reshape(n_samples, -1).T.astype(cp.float32)

        # One-hot encode labels
        y_batch = cp.zeros((self.model.output_dim, n_samples), dtype=cp.float32)
        y_batch[y_val.astype(int), cp.arange(n_samples)] = 1.0

        # Forward pass (no gradient computation needed)
        outputs = self.model.forward(X_batch)

        # Compute loss
        loss = -cp.sum(y_batch * cp.log(outputs + 1e-8)) / n_samples

        # Compute accuracy
        correct = cp.sum(cp.argmax(outputs, axis=0) == y_val)
        accuracy = float(correct / n_samples)

        return float(loss), accuracy

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
        """
        Full training loop.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of epochs to train
            batch_size: Batch size
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        for epoch in range(epochs):
            # Train one epoch
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size)

            # Validate
            val_loss, val_acc = self.validate(X_val, y_val)

            # Update learning rate if scheduler is provided
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch)

            # Check early stopping
            if self.early_stopping:
                if self.early_stopping.should_stop(val_loss):
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

        return self.history
