from ..layers import LayerType
from .base import BaseMLP
from .config import ModelConfig
from .factory import create_mlp
from .mlp import MLP

try:
    from .mlp_pytorch import MLPPyTorch, create_mlp_from_config

    _PYTORCH_AVAILABLE = True
except ImportError:
    MLPPyTorch = None
    create_mlp_from_config = None
    _PYTORCH_AVAILABLE = False

__all__ = [
    "BaseMLP",
    "MLP",
    "ModelConfig",
    "LayerType",
    "create_mlp",
]

if _PYTORCH_AVAILABLE:
    __all__.extend(["MLPPyTorch", "create_mlp_from_config"])
