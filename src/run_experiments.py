"""
Main script to run all experiments.

Usage:
    python src/run_experiments.py --experiment M0
    python src/run_experiments.py --experiment M1
    python src/run_experiments.py --all
"""

import argparse
from dataHandler import Dataset
from experiments import Experiment
from experiments.config import M0_CONFIG
from utils import ExperimentLogger, plot_training_history


def main():
    parser = argparse.ArgumentParser(description="Run MLP experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="M0",
        help="Experiment to run (M0, M1, M2, M3)",
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    args = parser.parse_args()

    # Load dataset (same split for all experiments)
    dataset = Dataset(
        X_path="data/X_images.npy",
        y_path="data/y_images.npy",
        dev_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        normalize=True,
        random_seed=42,
    )

    # Logger
    logger = ExperimentLogger()

    # Run experiments
    if args.all:
        configs = [M0_CONFIG]  # Add M1_CONFIG, M2_CONFIG, M3_CONFIG later
    else:
        # Map experiment name to config
        config_map = {
            "M0": M0_CONFIG,
            # 'M1': M1_CONFIG,
            # 'M2': M2_CONFIG,
            # 'M3': M3_CONFIG,
        }
        configs = [config_map[args.experiment]]

    # Run each experiment
    for config in configs:
        experiment = Experiment(config, dataset)
        results = experiment.run()

        # Log results
        logger.log_experiment(config, results)

        # Plot training history
        plot_training_history(
            results["history"], save_path=f"results/{config.name}_history.png"
        )


if __name__ == "__main__":
    main()
