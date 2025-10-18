# import numpy as cp
import cupy as cp
from tqdm import tqdm
from models import MLP


class MLPTrainer:
    def __init__(
        self,
        X_data: cp.ndarray,
        y_data: cp.ndarray,
        epochs: int,
        hidden_layers_neuron_count: list[int] = [64, 64],
        batch_size: int = 256,
        k_folds: int = 10,
    ):
        self.input_dim = X_data.shape[1] * X_data.shape[2]
        self.output_dim = len(cp.unique(y_data))
        self.model = MLP(
            batch_size,
            self.input_dim,
            self.output_dim,
            hidden_layers_neuron_count,
        )
        self.X_data = X_data
        self.y_data = y_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.k_folds = k_folds

    def data_splitting(self):
        # dividimos en folds
        # estratificamos los folds por defecto
        pass

    def train(self, X_train: cp.ndarray, y_train: cp.ndarray):
        batch_size = self.batch_size
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        sample_indices = cp.arange(n_samples)  # List of indices

        for epoch in tqdm(range(self.epochs)):
            # Fisher-Yates shuffle data
            cp.random.shuffle(sample_indices)
            X_trained_shuffled = X_train[sample_indices]
            y_trained_shuffled = y_train[sample_indices]

            epoch_correct = 0
            epoch_loss = 0.0

            # TODO : Hacer batch normalization
            for batch in range(n_batches):
                batch_start = batch * batch_size
                batch_end = min((batch + 1) * batch_size, n_samples)
                batch_data_X = X_trained_shuffled[batch_start:batch_end]
                batch_data_y = y_trained_shuffled[batch_start:batch_end]
                batch_length = batch_end - batch_start

                # (batch_length, 28, 28) -> (batch_length, 784) -> (784, batch_length)
                # labels -> (output_dim, batch_length)
                # TODO: The data should be linearized before training
                X_batch = batch_data_X.reshape(batch_length, -1).T.astype(cp.float32)
                y_batch = cp.zeros((self.output_dim, batch_length), dtype=cp.float32)
                y_batch[batch_data_y.astype(int), cp.arange(batch_length)] = 1.0

                # Forward pass
                outputs = self.model.forward(X_batch)
                # Compute loss
                loss = cp.clip(
                    -cp.sum(y_batch * cp.log(outputs + 1e-8)) / batch_length, 0, 1e8
                )
                epoch_loss += loss * batch_length  # Accumulate weighted loss
                # loss = self.model.loss(outputs, y_batch)
                # Average accuracy
                batch_correct = cp.sum(cp.argmax(outputs, axis=0) == batch_data_y)
                epoch_correct += batch_correct
                # Backward pass
                self.model.backward(y_batch)
                self.model.update_weights(learning_rate=0.001)

                # if (batch + 1) % 100 == 0:
                #    print(
                #        f"Batch {batch+1}/{n_batches}, Loss: {float(loss):.4f}, Accuracy: {float(batch_correct) / ((batch + 1) * batch_size):.4f}"
                #    )

            avg_loss = epoch_loss / n_samples
            accuracy = float(epoch_correct) / n_samples * 100
            tqdm.write(
                f"Epoch {epoch+1}/{self.epochs}, Avg Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%"
            )
        self.model.save_weights("weights.npz")


if __name__ == "__main__":
    X_images = cp.load("data/X_images.npy")
    y_images = cp.load("data/y_images.npy")
    trainer = MLPTrainer(X_images, y_images, epochs=10)
    trainer.train(X_images, y_images)
