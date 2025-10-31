import cupy as cp
import numpy as np
from numpy.typing import NDArray

from ..models.base import BaseMLP
from .metrics import compute_metrics


def evaluate_model(
    model: BaseMLP,
    X_test: np.ndarray | cp.ndarray,
    y_test: np.ndarray | cp.ndarray,
    include_confusion_matrix: bool = True,
    batch_size: int = 2048,
) -> dict[str, float | NDArray]:
    """Evaluate a model on test data using batched inference.

    X_test and y_test should be NumPy arrays (CPU).
    Batches are moved to GPU for prediction to avoid OOM errors.

    Args:
        model: Trained MLP model
        X_test: Test features (CPU, shape: [n_samples, n_features])
        y_test: Test labels (CPU, shape: [n_samples])
        include_confusion_matrix: Whether to compute confusion matrix
        batch_size: Batch size for inference (default 2048)
    """
    if isinstance(X_test, cp.ndarray):
        X_test = cp.asnumpy(X_test)
    if isinstance(y_test, cp.ndarray):
        y_test = cp.asnumpy(y_test)

    n_samples = X_test.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    all_predictions = []
    all_proba = []

    for batch in range(n_batches):
        batch_start = batch * batch_size
        batch_end = min((batch + 1) * batch_size, n_samples)

        X_batch_cpu = X_test[batch_start:batch_end]
        X_batch_gpu = cp.asarray(X_batch_cpu)
        y_pred_proba_batch = model.predict(X_batch_gpu.T)
        y_pred_batch = cp.argmax(y_pred_proba_batch, axis=0)

        all_predictions.append(y_pred_batch)
        all_proba.append(y_pred_proba_batch.T)

    y_pred = cp.concatenate(all_predictions)
    y_pred_proba = cp.concatenate(all_proba, axis=0)
    y_pred_proba = y_pred_proba.T

    y_test_gpu = cp.asarray(y_test)

    metrics = compute_metrics(
        y_test_gpu,
        y_pred,
        y_pred_proba,
        include_confusion_matrix=include_confusion_matrix,
    )

    return metrics


def compare_models(results: dict[str, dict[str, float]]) -> None:
    """
    Print a comparison table of multiple model results.
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    all_metrics: set[str] = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    all_metrics.discard("confusion_matrix")

    metric_names = sorted(all_metrics)
    print(f"\n{'Model':<20}", end="")
    for metric in metric_names:
        print(f"{metric.replace('_', ' ').title():<20}", end="")
    print()
    print("-" * (20 + 20 * len(metric_names)))

    for model_name, metrics in results.items():
        print(f"{model_name:<20}", end="")
        for metric in metric_names:
            value = metrics.get(metric, float("nan"))
            if isinstance(value, (int, float)):
                print(f"{value:<20.4f}", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()

    print()


def create_results_dict(
    model_evaluations: list[tuple[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """
    Convert a list of (model_name, metrics) tuples to a results dictionary.
    """
    return dict(model_evaluations)
