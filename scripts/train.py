"""
Train a model from a YAML experiment config, with optional CLI overrides.

Usage:
    python scripts/train.py --config configs/0_baselines/01_alphaearth_lightunet.yaml
    python scripts/train.py --config ... --epochs 1 --batch_size 4   # quick smoke run
"""
import argparse

from emb2heights.config import load_config
from emb2heights.trainers import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    # Overrides: only applied when explicitly passed (None means "use YAML value")
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--random_seed", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    config = load_config(args.config, overrides)
    print(f"Experiment: {config.experiment_name}")
    train(config)


if __name__ == "__main__":
    main()
