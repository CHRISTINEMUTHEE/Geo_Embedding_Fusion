"""Console script for emb2heights: `python -m emb2heights.cli --config <yaml>`."""
import argparse
import sys

from emb2heights.config import load_config
from emb2heights.trainers import train


def main():
    parser = argparse.ArgumentParser("Train emb2heights models")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    train(load_config(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
