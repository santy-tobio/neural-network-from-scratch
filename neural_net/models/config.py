from dataclasses import dataclass

from ..layers import LayerType


@dataclass
class ModelConfig:
    """
    Configuration for model architecture.

    Defines the structure and activation functions of the neural network.
    """

    hidden_layers: list[int]

    activation: LayerType = LayerType.RELU
    output_activation: LayerType = LayerType.SOFTMAX

    dropout_rate: float | None = None

    input_dim: int | None = None
    output_dim: int | None = None

    def __post_init__(self):
        assert len(self.hidden_layers) > 0, "Must have at least one hidden layer"
        assert all(
            units > 0 for units in self.hidden_layers
        ), "All layer sizes must be positive"

        if self.input_dim is not None:
            assert self.input_dim > 0, "Input dimension must be positive"

        if self.output_dim is not None:
            assert self.output_dim > 0, "Output dimension must be positive"

        if self.dropout_rate is not None:
            assert 0 < self.dropout_rate < 1, "Dropout rate must be in (0, 1)"

    def __repr__(self) -> str:
        return (
            f"ModelConfig(\n"
            f"  hidden_layers={self.hidden_layers},\n"
            f"  activation={self.activation.name},\n"
            f"  output_activation={self.output_activation.name},\n"
            f"  dropout_rate={self.dropout_rate},\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim}\n"
            f")"
        )
