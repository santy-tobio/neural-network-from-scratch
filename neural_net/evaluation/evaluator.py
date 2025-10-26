import cupy as cp

from ..models.base_mlp import BaseMLP

from .metrics import compute_metrics


class Evaluator:
    """Evaluates models on test data using consistent metrics."""

    def __init__(self):
        self.results = {}

    def evaluate(
        self, model: BaseMLP, X_test: cp.ndarray, y_test: cp.ndarray, model_name: str
    ) -> dict[str, float]:
        """Evaluate a model on test data."""
        y_pred_proba = model.predict(X_test)
        y_pred = cp.argmax(y_pred_proba, axis=0)

        metrics = compute_metrics(y_test, y_pred, y_pred_proba)
        self.results[model_name] = metrics

        return metrics

    def compare_models(self):
        """Compare all evaluated models."""
        # TODO: Create comparison table
        # Print accuracy, F1, cross-entropy for each model

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
            if "cross_entropy" in metrics:
                print(f"  Cross-Entropy: {metrics['cross_entropy']:.4f}")

        return self.results

    def evaluate_with_noise(self, model, X_test, y_test, noise_levels, model_name: str):
        """
        Evaluate model robustness to noise (for exercise 4d).

        Args:
            model: Model to evaluate
            X_test: Test data
            y_test: Test labels
            noise_levels: List of noise standard deviations to test
            model_name: Name of the model

        Returns:
            Dictionary with results for each noise level
        """
        results = {}

        for noise_std in noise_levels:
            # Add Gaussian noise
            X_noisy = X_test + cp.random.normal(0, noise_std, X_test.shape)
            X_noisy = cp.clip(X_noisy, 0, 1)  # Keep in valid range

            # Evaluate
            y_pred_proba = model.predict(X_noisy)
            y_pred = cp.argmax(y_pred_proba, axis=0)

            metrics = compute_metrics(y_test, y_pred, y_pred_proba)
            results[noise_std] = metrics

        return results
