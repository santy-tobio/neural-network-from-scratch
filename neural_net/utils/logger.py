"""
Logger for tracking experiments.
"""

import json
from datetime import datetime


class ExperimentLogger:
    """
    Logs experiment configurations and results.
    Useful for keeping track of all experiments.
    """

    def __init__(self, log_file: str = "experiments.log"):
        self.log_file = log_file

    def log_experiment(self, config, results):
        """
        Log an experiment.

        Args:
            config: ExperimentConfig object
            results: Dictionary with training history and test metrics
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "name": config.name,
            "description": config.description,
            "config": {
                "hidden_layers": config.hidden_layers,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "optimizer": config.optimizer,
            },
            "results": {
                "final_train_loss": results["history"]["train_loss"][-1],
                "final_val_loss": results["history"]["val_loss"][-1],
                "test_accuracy": results["test_metrics"]["accuracy"],
                "test_f1": results["test_metrics"]["f1_macro"],
            },
        }

        # Append to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"Experiment logged to {self.log_file}")
