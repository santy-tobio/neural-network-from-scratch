import cupy as cp


def accuracy(y_true, y_pred):
    """Compute accuracy"""
    return float(cp.mean(y_true == y_pred))


def cross_entropy(y_true, y_pred_proba, epsilon=1e-8):
    """
    Compute cross-entropy loss.

    Args:
        y_true: True labels (integers)
        y_pred_proba: Predicted probabilities (output_dim, n_samples)
        epsilon: Small constant for numerical stability
    """
    n_samples = y_true.shape[0]
    # Convert to one-hot if needed
    if len(y_pred_proba.shape) == 2:
        # y_pred_proba is (output_dim, batch_size) or (batch_size, output_dim)
        loss = -cp.sum(cp.log(y_pred_proba + epsilon)) / n_samples
    return float(loss)


def confusion_matrix_fn(y_true, y_pred):
    """
    Compute confusion matrix manually (no sklearn).

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix of shape (n_classes, n_classes)
    """
    # Get unique classes
    classes = cp.unique(cp.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    # Initialize confusion matrix
    cm = cp.zeros((n_classes, n_classes), dtype=cp.int32)

    # Fill confusion matrix
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = cp.sum((y_true == true_class) & (y_pred == pred_class))

    return cp.asnumpy(cm)  # Return as numpy for compatibility


def f1_score_macro(y_true, y_pred):
    """
    Compute macro F1 score manually (no sklearn).

    F1 = 2 * (precision * recall) / (precision + recall)
    Macro F1 = average of F1 scores for each class

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Macro F1 score
    """
    classes = cp.unique(y_true)
    f1_scores = []

    for class_label in classes:
        # True positives
        tp = cp.sum((y_true == class_label) & (y_pred == class_label))

        # False positives
        fp = cp.sum((y_true != class_label) & (y_pred == class_label))

        # False negatives
        fn = cp.sum((y_true == class_label) & (y_pred != class_label))

        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 score for this class
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(float(f1))

    # Return macro average
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Compute all metrics at once.
    """
    metrics = {
        "accuracy": accuracy(y_true, y_pred),
        "f1_macro": f1_score_macro(y_true, y_pred),
        "confusion_matrix": confusion_matrix_fn(y_true, y_pred),
    }

    if y_pred_proba is not None:
        metrics["cross_entropy"] = cross_entropy(y_true, y_pred_proba)

    return metrics
