"""
Results Manager - Save and load experiment results

Handles saving and loading of:
- Training history (train/val loss and metrics)
- Test metrics (accuracy, F1, cross-entropy, confusion matrix)
- Model weights
- Experiment configuration
"""

import json
import time
from pathlib import Path

import cupy as cp
import numpy as np


class ResultsManager:
    """Manages saving and loading of experiment results."""

    def __init__(self, results_dir: str | Path = "results"):
        """
        Initialize ResultsManager.

        Args:
            results_dir: Base directory to save results (will create model subdirs)
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        print(f"Results will be saved to: {self.results_dir}")

    def save_experiment(
        self,
        model_name: str,
        config,
        history: dict,
        test_metrics: dict | None,
        model,
        train_time: float,
    ):
        """
        Save complete experiment results.

        Args:
            model_name: Name of the model (e.g., "M0", "M1a", "M1b")
            config: ExperimentConfig object
            history: Training history dict with train_loss, val_loss, etc.
            test_metrics: Dict with accuracy, f1_macro, cross_entropy, confusion_matrix (optional)
            model: Trained model object
            train_time: Training time in seconds
        """
        print(f"\n💾 Saving results for {model_name}...")

        # Determine model directory structure
        # M0 -> results/M0/
        # M1a -> results/M1/M1a/
        # M2 -> results/M2/
        if model_name.startswith("M1") and len(model_name) > 2:
            # M1 variants: M1a, M1b, etc.
            model_dir = self.results_dir / "M1" / model_name
        else:
            # Base models: M0, M2, M3
            model_dir = self.results_dir / model_name

        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"  📁 Model directory: {model_dir}")

        metadata = {
            "name": str(model_name),
            "description": str(config.description),
            "config": {
                "hidden_layers": list(config.model.hidden_layers),
                "dropout_rate": (
                    float(config.model.dropout_rate)
                    if config.model.dropout_rate is not None
                    else 0.0
                ),
                "optimizer": str(config.training.optimizer.type.value),
                "learning_rate": float(config.training.optimizer.learning_rate),
                "momentum": (
                    float(config.training.optimizer.momentum)
                    if hasattr(config.training.optimizer, "momentum")
                    and config.training.optimizer.momentum is not None
                    else 0.0
                ),
                "batch_size": int(config.training.batch_size),
                "epochs": int(config.training.epochs),
                "use_l2": bool(config.training.regularizer.use_l2),
                "l2_lambda": (
                    float(config.training.regularizer.l2_lambda)
                    if config.training.regularizer.use_l2
                    and config.training.regularizer.l2_lambda is not None
                    else 0.0
                ),
                "early_stopping": bool(config.training.early_stopping.enabled),
            },
            "test_metrics": (
                {
                    "accuracy": float(test_metrics["accuracy"]),
                    "f1_macro": float(test_metrics["f1_macro"]),
                    "cross_entropy": float(test_metrics["cross_entropy"]),
                }
                if test_metrics is not None
                else None
            ),
            "train_time_seconds": float(train_time),
            "epochs_completed": int(len(history["train_loss"])),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        json_path = model_dir / f"{model_name.lower()}_results.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Saved metadata: {json_path.name}")

        # 2. Save training history (NPZ)
        history_data = {
            "train_loss": np.array(history["train_loss"]),
            "train_metric": np.array(history["train_metric"]),
            "val_loss": np.array(history["val_loss"]),
            "val_metric": np.array(history["val_metric"]),
        }

        history_path = model_dir / f"{model_name.lower()}_history.npz"
        np.savez_compressed(history_path, **history_data)
        print(f"  ✓ Saved training history: {history_path.name}")

        # 3. Save confusion matrix (NPY) - only if test_metrics provided
        if test_metrics is not None:
            cm = test_metrics["confusion_matrix"]
            # Convert to numpy if it's a CuPy array
            if isinstance(cm, cp.ndarray):
                cm = cp.asnumpy(cm)

            cm_path = model_dir / f"{model_name.lower()}_confusion_matrix.npy"
            np.save(cm_path, cm)
            print(f"  ✓ Saved confusion matrix: {cm_path.name}")

        # 4. Save model weights (NPZ)
        weights = model.get_parameters()
        weights_dict = {}

        for i, layer_weights in enumerate(weights):
            # Convert CuPy arrays to NumPy
            if isinstance(layer_weights, dict):
                for key, value in layer_weights.items():
                    if isinstance(value, cp.ndarray):
                        value = cp.asnumpy(value)
                    weights_dict[f"layer_{i}_{key}"] = value
            elif isinstance(layer_weights, cp.ndarray):
                weights_dict[f"layer_{i}"] = cp.asnumpy(layer_weights)

        weights_path = model_dir / f"{model_name.lower()}_weights.npz"
        np.savez_compressed(weights_path, **weights_dict)
        print(f"  ✓ Saved model weights: {weights_path.name}")

        print(f"✓ All results saved for {model_name}\n")

    def load_results(self, model_name: str) -> dict:
        """
        Load experiment results.

        Args:
            model_name: Name of the model (e.g., "M0", "M1a", "M1b")

        Returns:
            Dict with all loaded data
        """
        print(f"\n📂 Loading results for {model_name}...")

        # Determine model directory
        if model_name.startswith("M1") and len(model_name) > 2:
            model_dir = self.results_dir / "M1" / model_name
        else:
            model_dir = self.results_dir / model_name

        results = {}

        # Load metadata
        json_path = model_dir / f"{model_name.lower()}_results.json"
        with open(json_path) as f:
            results["metadata"] = json.load(f)
        print(f"  ✓ Loaded metadata")

        # Load training history
        history_path = model_dir / f"{model_name.lower()}_history.npz"
        history_data = np.load(history_path)
        results["history"] = {
            "train_loss": history_data["train_loss"],
            "train_metric": history_data["train_metric"],
            "val_loss": history_data["val_loss"],
            "val_metric": history_data["val_metric"],
        }
        print(f"  ✓ Loaded training history")

        # Load confusion matrix
        cm_path = model_dir / f"{model_name.lower()}_confusion_matrix.npy"
        results["confusion_matrix"] = np.load(cm_path)
        print(f"  ✓ Loaded confusion matrix")

        # Load weights
        weights_path = model_dir / f"{model_name.lower()}_weights.npz"
        results["weights"] = np.load(weights_path)
        print(f"  ✓ Loaded model weights")

        print(f"✓ All results loaded for {model_name}\n")
        return results

    def list_experiments(self) -> list[str]:
        """List all saved experiments."""
        experiments = []

        # Find all JSON result files recursively
        for json_file in self.results_dir.rglob("*_results.json"):
            model_name = json_file.stem.replace("_results", "").upper()
            experiments.append(model_name)

        return sorted(experiments)

    def compare_experiments(self, model_names: list[str]) -> None:
        """
        Print comparison table of multiple experiments.

        Args:
            model_names: List of model names to compare
        """
        print("\n" + "=" * 100)
        print("COMPARISON TABLE")
        print("=" * 100)

        # Header
        print(
            f"{'Model':<10} {'Accuracy':<12} {'F1-Macro':<12} {'Cross-Ent':<12} "
            f"{'Train Time':<15} {'Epochs':<10}"
        )
        print("-" * 100)

        # Load and display each model
        for model_name in model_names:
            # Determine model directory
            if model_name.startswith("M1") and len(model_name) > 2:
                model_dir = self.results_dir / "M1" / model_name
            else:
                model_dir = self.results_dir / model_name

            json_path = model_dir / f"{model_name.lower()}_results.json"

            if not json_path.exists():
                print(f"{model_name:<10} (not found)")
                continue

            with open(json_path) as f:
                data = json.load(f)

            metrics = data["test_metrics"]
            print(
                f"{model_name:<10} "
                f"{metrics['accuracy']:<12.4f} "
                f"{metrics['f1_macro']:<12.4f} "
                f"{metrics['cross_entropy']:<12.4f} "
                f"{data['train_time_seconds']:<15.2f} "
                f"{data['epochs_completed']:<10}"
            )

        print("=" * 100)
