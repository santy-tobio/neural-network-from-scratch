import cupy as cp

from ..layers import Dropout, Layer, LayerType, Linear
from .base import BaseMLP


class MLP(BaseMLP):
    """
    Multi-Layer Perceptron with configurable hidden layers and activations.

    Architecture: Input -> [Linear -> Activation] * n_hidden -> Linear -> Output Activation
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        activation: LayerType = LayerType.RELU,
        output_activation: LayerType = LayerType.SOFTMAX,
        dropout_rate: float | None = None,
    ):
        super().__init__(input_dim, output_dim, hidden_layers)

        self.layers: list[Layer] = []

        layer_sizes = [input_dim] + hidden_layers

        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers.append(activation.value())

            if dropout_rate is not None:
                self.layers.append(Dropout(dropout_rate))

        self.layers.append(Linear(layer_sizes[-1], output_dim))
        self.layers.append(output_activation.value())

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

    def get_parameters(self):
        """
        Return current parameters for saving.
        Returns: list of dicts with 'weights' and 'bias' for each Linear layer
        """
        parameters = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                parameters.append({"weights": layer.weights, "bias": layer.bias})
        return parameters

    def update_parameters(self, updates):
        """
        Update parameters given updates from optimizer.
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
        activations = X
        for layer in self.layers:
            activations = layer.evaluate(activations)
        return activations
