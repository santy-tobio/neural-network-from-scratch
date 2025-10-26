import cupy as cp
from numpy.typing import NDArray

from ..models.base import BaseMLP
from .metrics import compute_metrics


def evaluate_model(
    model: BaseMLP,
    X_test: cp.ndarray,
    y_test: cp.ndarray,
    include_confusion_matrix: bool = True,
) -> dict[str, float | NDArray]:
    """Evaluate a model on test data."""
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = cp.argmax(y_pred_proba, axis=0)

    # Compute metrics
    metrics = compute_metrics(
        y_test, y_pred, y_pred_proba, include_confusion_matrix=include_confusion_matrix
    )

    return metrics


def compare_models(results: dict[str, dict[str, float]]) -> None:
    """
    Print a comparison table of multiple model results.
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # Determine which metrics are available
    all_metrics: set[str] = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    all_metrics.discard("confusion_matrix")  # Don't print in comparison

    # Print header
    metric_names = sorted(all_metrics)
    print(f"\n{'Model':<20}", end="")
    for metric in metric_names:
        print(f"{metric.replace('_', ' ').title():<20}", end="")
    print()
    print("-" * (20 + 20 * len(metric_names)))

    # Print each model's results
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
