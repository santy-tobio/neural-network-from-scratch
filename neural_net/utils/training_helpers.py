import cupy as cp


def prepare_classification_batch(
    X_raw: cp.ndarray, y_raw: cp.ndarray, output_dim: int
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Standard preprocessing for classification tasks.
    Flattens input and one-hot encodes labels.
    """
    batch_size = X_raw.shape[0]

    # Flatten and transpose: (batch_size, ...) -> (input_dim, batch_size)
    X_batch = X_raw.reshape(batch_size, -1).T.astype(cp.float32)

    # One-hot encode labels: (output_dim, batch_size)
    y_onehot = cp.zeros((output_dim, batch_size), dtype=cp.float32)
    y_onehot[y_raw.astype(int), cp.arange(batch_size)] = 1.0

    return X_batch, y_onehot
