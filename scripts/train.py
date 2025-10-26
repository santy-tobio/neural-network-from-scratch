import argparse

from experiments import Experiment, configs
from neural_net.data import Dataset
from neural_net.utils import ExperimentLogger, plot_training_history


def main():
    parser = argparse.ArgumentParser(description="Run MLP experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="M0",
        help="Experiment to run",
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    args = parser.parse_args()

    # Load dataset (same split for all experiments)
    dataset = Dataset(
        X_path="data/X_images.npy",
        y_path="data/y_images.npy",
        dev_ratio=0.85,  # 85% for dev (train+val), 15% for test
        normalize=True,
        random_seed=42,
    )

    # Logger
    logger = ExperimentLogger()
    all_configs_list = [
        getattr(configs, name)
        for name in dir(configs)
        if name.endswith("_CONFIG") and not name.startswith("_")
    ]

    # Run experiments
    if args.all:
        configs_list = all_configs_list
    else:
        # Dynamically get config by name
        config_name = f"{args.experiment.upper()}_CONFIG"
        try:
            configs_list = [getattr(configs, config_name)]
        except AttributeError:
            print(f"Error: Experiment '{args.experiment}' not found.")
            print(f"Available experiments: {all_configs_list}")
            return

    # Run each experiment
    for config in configs_list:
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
