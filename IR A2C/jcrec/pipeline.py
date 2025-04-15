import os
import argparse
import yaml
import torch
import numpy as np
import random

from Dataset import Dataset
from Greedy import Greedy
from Optimal import Optimal
from Reinforce import Reinforce


def set_seed(seed=42):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior


def create_and_print_dataset(config):
    """Create and print the dataset."""
    dataset = Dataset(config)
    print(dataset)
    return dataset


def main():
    """Run the recommender system based on the provided model and parameters."""
    parser = argparse.ArgumentParser(description="Run recommender models.")
    parser.add_argument(
        "--config", help="Path to the configuration file", default="/home/jrajend/IR A2C/config/run.yaml"
    )

    args = parser.parse_args()

    # Set the seed before initializing anything
    set_seed(42)

    # Ensure the config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Debugging output to confirm correct config file is read
    print(f"Loaded config from: {args.config}")
    print(f"Using model: {config.get('model', 'Unknown')}")

    model_classes = {
        "greedy": Greedy,
        "optimal": Optimal,
        "reinforce": Reinforce,
    }

    for run in range(config["nb_runs"]):
        dataset = create_and_print_dataset(config)

        # If the model is Greedy or Optimal, use respective classes
        if config["model"] in ["greedy", "optimal"]:
            recommender = model_classes[config["model"]](dataset, config["threshold"])
            recommendation_method = getattr(
                recommender, f'{config["model"]}_recommendation'
            )
            recommendation_method(config["k"], run)

        # Use Reinforce (A2C) for RL-based recommendation
        elif config["model"] == "a2c":
            print(f"Running A2C model for run {run + 1}/{config['nb_runs']}...")  # ✅ Debugging log
            recommender = Reinforce(
                dataset,
                config["model"],
                config["k"],
                config["threshold"],
                run,
                config["total_steps"],
                config["eval_freq"],
            )
            recommender.reinforce_recommendation()

        else:
            raise ValueError(f"Unsupported model type: {config['model']}")  # ✅ Error handling


if __name__ == "__main__":
    main()
