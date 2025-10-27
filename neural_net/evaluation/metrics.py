from enum import Enum

import cupy as cp
import numpy as np
from numpy.typing import NDArray


def _macro_average(values: list[float], _: list[float] | None = None) -> float:
    """Compute macro average (unweighted mean)."""
    return sum(values) / len(values) if values else 0.0


def _weighted_average(values: list[float], weights: list[float] | None = None) -> float:
    """Compute weighted average."""
    if weights is None:
        return _macro_average(values)
    total_weight = sum(weights)
    if total_weight > 0:
        return sum(v * w for v, w in zip(values, weights, strict=False)) / total_weight
    return 0.0


class AverageStrategy(Enum):
    """Averaging strategies for multi-class metrics."""

    MACRO = _macro_average
    WEIGHTED = _weighted_average


def accuracy(y_true: cp.ndarray, y_pred: cp.ndarray) -> float:
    """Compute classification accuracy."""
    return float(cp.mean(y_true == y_pred))


def cross_entropy(
    y_true: cp.ndarray, y_pred_proba: cp.ndarray, epsilon: float = 1e-8
) -> float:
    """Compute cross-entropy loss for multi-class classification."""
    n_samples = y_true.shape[0]

    # Ensure y_pred_proba is (n_classes, n_samples)
    if y_pred_proba.shape[1] != n_samples:
        y_pred_proba = y_pred_proba.T

    # Extract the predicted probability for the true class for each sample
    # y_pred_proba[y_true[i], i] gives the probability of the correct class for sample i
    true_class_probs = y_pred_proba[y_true, cp.arange(n_samples)]

    # Clip probabilities for numerical stability
    true_class_probs = cp.clip(true_class_probs, epsilon, 1.0 - epsilon)

    # Compute cross-entropy: -mean(log(p_correct))
    loss = -cp.mean(cp.log(true_class_probs))

    return float(loss)


def confusion_matrix(y_true: cp.ndarray, y_pred: cp.ndarray) -> NDArray[np.int32]:
    """Compute confusion matrix."""
    # Get unique classes from both arrays
    classes = cp.unique(cp.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    # Initialize confusion matrix
    cm = cp.zeros((n_classes, n_classes), dtype=cp.int32)

    # Fill confusion matrix
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = cp.sum((y_true == true_class) & (y_pred == pred_class))

    return cp.asnumpy(cm)


def precision_recall_f1(
    y_true: cp.ndarray,
    y_pred: cp.ndarray,
    average: AverageStrategy = AverageStrategy.MACRO,
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 score."""
    classes = cp.unique(y_true)
    precisions = []
    recalls = []
    f1_scores = []
    weights = []

    for class_label in classes:
        # True positives, false positives, false negatives
        tp = float(cp.sum((y_true == class_label) & (y_pred == class_label)))
        fp = float(cp.sum((y_true != class_label) & (y_pred == class_label)))
        fn = float(cp.sum((y_true == class_label) & (y_pred != class_label)))

        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        weights.append(tp + fn)  # Number of true instances
        # The enum is directly callable
        avg_precision = average(precisions, weights)
        avg_recall = average(recalls, weights)
        avg_f1 = average(f1_scores, weights)

    return avg_precision, avg_recall, avg_f1


def f1_score(
    y_true: cp.ndarray,
    y_pred: cp.ndarray,
    average: AverageStrategy = AverageStrategy.MACRO,
) -> float:
    """
    Compute F1 score.

    F1 = 2 * (precision * recall) / (precision + recall)
    """
    _, _, f1 = precision_recall_f1(y_true, y_pred, average=average)
    return f1


def compute_metrics(
    y_true: cp.ndarray,
    y_pred: cp.ndarray,
    y_pred_proba: cp.ndarray | None = None,
    include_confusion_matrix: bool = True,
) -> dict[str, float | NDArray[np.int32]]:
    """Compute multiple classification metrics at once."""
    metrics: dict[str, float | NDArray[np.int32]] = {
        "accuracy": accuracy(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average=AverageStrategy.MACRO),
    }

    if include_confusion_matrix:
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    if y_pred_proba is not None:
        metrics["cross_entropy"] = cross_entropy(y_true, y_pred_proba)

    return metrics
