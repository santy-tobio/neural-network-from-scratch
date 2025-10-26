from enum import Enum
from typing import Any

import cupy as cp
from numpy.typing import NDArray

from ..models.base import BaseMLP
from .metrics import compute_metrics


class NoiseType(Enum):
    GAUSSIAN = cp.random.normal
    UNIFORM = cp.random.uniform


def evaluate_with_noise(
    model: BaseMLP,
    X_test: cp.ndarray,
    y_test: cp.ndarray,
    noise_levels: list[float],
    noise_type: NoiseType = NoiseType.GAUSSIAN,
    clip_range: tuple[float, float] = (0.0, 1.0),
) -> dict[float, dict[str, Any]]:
    """Evaluate model robustness to input noise."""
    results: dict[float, dict[str, Any]] = {}

    for noise_std in noise_levels:
        # Add noise based on type
        X_noisy = noise_type.value(0, noise_std, X_test.shape) + X_test

        # Clip to valid range
        X_noisy = cp.clip(X_noisy, clip_range[0], clip_range[1])

        # Evaluate
        y_pred_proba = model.predict(X_noisy)
        y_pred = cp.argmax(y_pred_proba, axis=0)

        metrics = compute_metrics(
            y_test, y_pred, y_pred_proba, include_confusion_matrix=False
        )
        results[noise_std] = metrics

    return results


def evaluate_with_dropout(
    model: BaseMLP,
    X_test: cp.ndarray,
    y_test: cp.ndarray,
    n_samples: int = 10,
) -> dict[str, float | NDArray]:
    """Evaluate model with dropout enabled (Monte Carlo dropout)."""
    all_predictions = []

    # Enable dropout during inference
    original_training = model.training
    model.training = True

    try:
        for _ in range(n_samples):
            y_pred_proba = model.predict(X_test)
            all_predictions.append(y_pred_proba)
    finally:
        # Restore original training state
        model.training = original_training

    # Stack predictions and compute statistics
    all_predictions = cp.stack(all_predictions, axis=0)  # (n_samples, n_classes, batch)
    mean_proba = cp.mean(all_predictions, axis=0)  # (n_classes, batch)
    std_proba = cp.std(all_predictions, axis=0)  # (n_classes, batch)

    # Get final predictions
    y_pred = cp.argmax(mean_proba, axis=0)

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, mean_proba, include_confusion_matrix=True)

    # Add uncertainty metrics
    metrics["prediction_std"] = float(cp.mean(std_proba))
    metrics["prediction_entropy"] = float(
        cp.mean(-cp.sum(mean_proba * cp.log(mean_proba + 1e-8), axis=0))
    )

    return metrics
