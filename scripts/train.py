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
    parser.add_argument("--loss_name", type=str, help="mae | weighted (see emb2heights/losses.py)")
    parser.add_argument("--w_height", type=float)
    parser.add_argument("--w_landcover", type=float, help="0.0 ablates the auxiliary landcover head (RQ3)")
    parser.add_argument("--bg_weight", type=float)
    parser.add_argument("--max_train_tiles", type=int,
                        help="Cap training tiles (val stays full) -- for a label-efficiency sweep, see scripts/label_efficiency_sweep.py")
    parser.add_argument("--iou_threshold", type=float,
                        help="Presence cutoff for building/vegetation/water IoU (decoupled from --height_mask_threshold)")
    parser.add_argument("--height_mask_threshold", type=float,
                        help="Presence cutoff for which pixels count toward rmse_building/rmse_vegetation")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    config = load_config(args.config, overrides)
    print(f"Experiment: {config.experiment_name}")
    train(config)


if __name__ == "__main__":
    main()
