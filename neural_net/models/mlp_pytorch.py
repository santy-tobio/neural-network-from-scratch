"""
PyTorch-based Multi-Layer Perceptron implementation.

This module provides PyTorch implementations of MLP models for comparison
with the custom CuPy/NumPy implementation.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MLPPyTorch(nn.Module):
    """
    Multi-Layer Perceptron implemented in PyTorch.

    Mirrors the architecture of the custom MLP implementation but uses
    PyTorch's automatic differentiation and optimized operations.

    Args:
        input_dim: Number of input features (784 for EMNIST)
        hidden_layers: List of hidden layer sizes
        output_dim: Number of output classes (47 for EMNIST)
        activation: Activation function ('relu', 'leaky_relu', 'gelu', 'silu', 'swish')
        dropout_rate: Dropout probability (0.0 = no dropout)
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super(MLPPyTorch, self).__init__()

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(self._get_activation(activation))

            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "swish": nn.SiLU(),
        }

        if activation.lower() not in activations:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Available: {list(activations.keys())}"
            )

        return activations[activation.lower()]

    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU-like activations."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits tensor of shape (batch_size, output_dim)
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions (argmax of logits).

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Predicted class indices of shape (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        return {
            "input_dim": self.input_dim,
            "hidden_layers": self.hidden_layers,
            "output_dim": self.output_dim,
            "activation": self.activation_name,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mlp_from_config(
    input_dim: int,
    output_dim: int,
    hidden_layers: List[int],
    activation: str = "relu",
    dropout_rate: float = 0.0,
    use_batch_norm: bool = False,
) -> MLPPyTorch:
    """
    Factory function to create MLP from configuration.

    Args:
        input_dim: Number of input features
        output_dim: Number of output classes
        hidden_layers: List of hidden layer sizes
        activation: Activation function name
        dropout_rate: Dropout probability
        use_batch_norm: Whether to use batch normalization

    Returns:
        Initialized MLPPyTorch model
    """
    return MLPPyTorch(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        activation=activation,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
    )
