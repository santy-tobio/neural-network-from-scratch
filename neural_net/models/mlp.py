import cupy as cp
from ..layers import CESoftmax, Linear, Relu
from .base_mlp import BaseMLP


class MLP(BaseMLP):
    """
    Multi-Layer Perceptron with configurable hidden layers.

    Architecture: Input -> [Linear -> ReLU] * n_hidden -> Linear -> Softmax
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_layers: list[int], batch_size: int
    ):
        super().__init__(input_dim, output_dim, hidden_layers, batch_size)

        self.batch_size = batch_size
        self.layers = []

        # Build hidden layers: [input_dim] + hidden_layers
        layer_sizes = [input_dim] + hidden_layers

        # Add hidden layers with ReLU activation
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1], batch_size))
            self.layers.append(Relu())

        # Add output layer with Softmax
        self.layers.append(Linear(layer_sizes[-1], output_dim, batch_size))
        self.layers.append(CESoftmax())

    def forward(self, X):
        """Forward pass through the network"""
        activations = X
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def backward(self, y_true):
        """Backward pass to compute gradients"""
        grad = y_true
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_gradients(self):
        """
        Return current gradients for optimizer.
        Returns: list of (grad_weights, grad_bias) tuples for each layer
        """
        gradients = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                gradients.append((layer.grad_weights, layer.grad_bias))
        return gradients

    def update_parameters(self, updates):
        """
        Update parameters given updates from optimizer.
        Args:
            updates: list of (weight_update, bias_update) tuples
        """
        update_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                weight_update, bias_update = updates[update_idx]
                layer.weights -= weight_update
                layer.bias -= bias_update
                update_idx += 1

    def save_weights(self, filepath: str):
        """Save model weights to file"""
        weights = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                weights.append(layer.weights.get())
                weights.append(layer.bias.get())
        cp.savez(filepath, *weights)

    def load_weights(self, filepath: str):
        """Load model weights from file"""
        weights_data = cp.load(filepath)
        linear_layer_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                w = weights_data[f"arr_{linear_layer_idx * 2}"]
                b = weights_data[f"arr_{linear_layer_idx * 2 + 1}"]
                layer.weights = cp.asarray(w)
                layer.bias = cp.asarray(b)
                linear_layer_idx += 1

    def predict(self, X):
        """Make predictions"""
        return self.forward(X)
