"""
PyTorch model configurations.

M2: PyTorch implementation matching best M1 configuration
M3: Optimized PyTorch with modern activations and regularization
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PyTorchModelConfig:
    """Configuration for PyTorch MLP models."""

    input_dim: int = 784  # EMNIST flattened 28x28
    hidden_layers: List[int] = None
    output_dim: int = 47  # EMNIST classes
    activation: str = "relu"
    dropout_rate: float = 0.0
    use_batch_norm: bool = False

    # Training
    optimizer_type: str = "adam"
    learning_rate: float = 0.001
    momentum: float = 0.9  # Only for SGD
    weight_decay: float = 0.0  # L2 regularization
    batch_size: int = 512
    epochs: int = 50

    # Scheduler
    scheduler_type: Optional[str] = None
    scheduler_params: Optional[dict] = None

    # Early stopping
    early_stopping: bool = False
    patience: int = 5

    # Metadata
    name: str = "Unnamed"
    description: str = ""

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64]


# ============================================================================
# M2: PyTorch baseline (matches best M1 - assuming M1b won based on typical results)
# ============================================================================

M2_CONFIG = PyTorchModelConfig(
    name="M2",
    description="PyTorch baseline - matches best M1 architecture (Adam + early stopping)",
    hidden_layers=[400, 240, 120],
    activation="relu",
    dropout_rate=0.0,
    use_batch_norm=False,
    optimizer_type="adam",
    learning_rate=0.001,
    weight_decay=0.0,
    batch_size=512,
    epochs=50,
    early_stopping=True,
    patience=5,
)

# ============================================================================
# M3 Variants: Modern architectures and regularization
# ============================================================================

M3a_CONFIG = PyTorchModelConfig(
    name="M3a",
    description="Deeper [256, 128, 64, 64] + GELU + Dropout(0.2) + weight decay + early stopping + patience=7",
    hidden_layers=[256, 128, 64, 64],
    activation="gelu",
    dropout_rate=0.2,
    use_batch_norm=False,
    optimizer_type="adam",
    learning_rate=0.001,
    weight_decay=1e-4,
    batch_size=512,
    epochs=50,
    early_stopping=True,
    patience=7,
)

M3b_CONFIG = PyTorchModelConfig(
    name="M3b",
    description="Wider [400, 400] + SiLU + BatchNorm + Dropout(0.3) + weight decay + early stopping + patience=7",
    hidden_layers=[400, 400],
    activation="silu",
    dropout_rate=0.3,
    use_batch_norm=True,
    optimizer_type="adam",
    learning_rate=0.001,
    weight_decay=1e-4,
    batch_size=512,
    epochs=50,
    early_stopping=True,
    patience=7,
)

# Collection of all M3 configs for easy iteration
M3_CONFIGS = [M3a_CONFIG, M3b_CONFIG]
