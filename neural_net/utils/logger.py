import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """
    Logs experiment configurations and results.
    Useful for keeping track of all experiments.
    """

    def __init__(self, log_file: str | Path = "experiments.log") -> None:
        """Initialize the experiment logger."""
        self.log_file = Path(log_file)
        # Create parent directories if they don't exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_experiment(
        self,
        config: Any,
        results: dict[str, Any],
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """Log an experiment."""
        # Build log entry
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "name": config.name,
            "description": config.description,
            "config": repr(config),
            "results": self._extract_results(results),
        }

        # Add extra info if provided
        if extra_info:
            log_entry["extra_info"] = extra_info

        # Append to log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, indent=2) + "\n")

        print(f"Experiment '{config.name}' logged to {self.log_file}")

    def _extract_results(self, results: dict[str, Any]) -> dict[str, Any]:
        """Extract and format results for logging."""
        extracted: dict[str, Any] = {}

        # Extract training history (last values)
        if "history" in results:
            history = results["history"]
            extracted["final_train_loss"] = (
                float(history["train_loss"][-1]) if history["train_loss"] else None
            )
            extracted["final_val_loss"] = (
                float(history["val_loss"][-1]) if history["val_loss"] else None
            )
            extracted["final_train_acc"] = (
                float(history["train_acc"][-1])
                if "train_acc" in history and history["train_acc"]
                else None
            )
            extracted["final_val_acc"] = (
                float(history["val_acc"][-1])
                if "val_acc" in history and history["val_acc"]
                else None
            )
            extracted["epochs_trained"] = len(history["train_loss"])

        # Extract test metrics
        if "test_metrics" in results:
            test_metrics = results["test_metrics"]
            extracted["test_accuracy"] = float(test_metrics.get("accuracy", 0))
            extracted["test_f1_macro"] = float(test_metrics.get("f1_macro", 0))
            if "cross_entropy" in test_metrics:
                extracted["test_cross_entropy"] = float(test_metrics["cross_entropy"])

        return extracted

    def load_logs(self) -> list[dict[str, Any]]:
        """
        Load all logged experiments.

        Returns:
            List of experiment log entries
        """
        if not self.log_file.exists():
            return []

        logs: list[dict[str, Any]] = []
        with open(self.log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))

        return logs

    def get_best_experiment(
        self, metric: str = "test_accuracy"
    ) -> dict[str, Any] | None:
        """Get the best experiment based on a specific metric."""
        logs = self.load_logs()
        if not logs:
            return None

        # Filter logs that have the metric
        valid_logs = [log for log in logs if metric in log.get("results", {})]
        if not valid_logs:
            return None

        # Return the one with the highest metric value
        return max(valid_logs, key=lambda x: x["results"][metric])

    def summary(self) -> None:
        """Print a summary of all logged experiments."""
        logs = self.load_logs()
        if not logs:
            print("No experiments logged yet.")
            return

        print(f"\n{'='*80}")
        print(f"EXPERIMENT SUMMARY ({len(logs)} experiments)")
        print(f"{'='*80}\n")

        for i, log in enumerate(logs, 1):
            print(f"{i}. {log['name']}: {log['description']}")
            results = log.get("results", {})
            if "test_accuracy" in results:
                print(f"   Test Accuracy: {results['test_accuracy']:.4f}")
            if "test_f1_macro" in results:
                print(f"   Test F1: {results['test_f1_macro']:.4f}")
            print(f"   Timestamp: {log['timestamp']}")
            print()
