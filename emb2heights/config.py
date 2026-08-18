"""
Experiment configuration.

One pydantic model = one experiment. Values come from a YAML file in configs/,
optionally overridden by CLI flags (see scripts/train.py). Derived paths are
computed properties so they always follow experiment_name.
"""
from enum import Enum
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel


## Closed set of choices only. Adding a model = value here + branch in models.build_model.
class ModelNameEnum(str, Enum):
    lightunet = "lightunet"
    efficientdecoder = "efficientdecoder"


class ExperimentConfig(BaseModel):
    # Experiment identity
    experiment_name: str
    base_dir: str = "outputs"

    # Data locations (embeddings + matching label rasters) — used by the legacy
    # random-split loader in datasets.py (build_dataloaders)
    train_embeddings_dir: Optional[str] = None
    train_targets_dir: Optional[str] = None
    test_embeddings_dir: Optional[str] = None

    # Region-grouped data loading (emb2heights.datamodule.Embed2HeightsDataModule).
    # When data_root is set, trainers.train() uses this instead of build_dataloaders —
    # it dequantizes int8 embeddings and masks nodata honestly (see datamodule.py).
    data_root: Optional[str] = None
    embedding_source: str = "alphaearth"

    # Model
    model_name: ModelNameEnum = ModelNameEnum.lightunet
    ## n_channels is NOT configured: it is inferred from the embedding files at
    ## runtime (AlphaEarth=64, Tessera=128, ...), so config can't disagree with data.
    n_classes: int = 4  # building %, vegetation %, water %, nDSM height
    height_normalization_constant: float = 30.0  # heights are meters, typical max ~30
    
    # Training
    batch_size: int = 32
    patch_size: int = 128
    num_workers: int = 4
    epochs: int = 30
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    val_split: float = 0.2
    random_seed: int = 42

    # Optimizer / scheduler
    optimizer: str = "adam"          # adam | adamw | sgd
    scheduler: str = "plateau"       # plateau | cosine | step | none
    patience: int = 10               # plateau
    factor: float = 0.1              # plateau
    step_size: int = 30              # step
    gamma: float = 0.1               # step

    # Loss weights [MAE, SSIM, Gradient, Tversky] — only MAE used until losses.py grows
    lambdas: List[float] = [1.0, 0.5, 0.5, 2.0]
    # Loss
    loss_name:str = "mae"
    ## Derived paths: everything lands under outputs/<experiment_name>/
    @property
    def experiment_dir(self) -> Path:
        return Path(self.base_dir) / self.experiment_name

    @property
    def viz_output_dir(self) -> Path:
        return self.experiment_dir / "visualizations"

    @property
    def best_model_path(self) -> Path:
        return self.experiment_dir / "best_model.pth"

    @property
    def last_model_path(self) -> Path:
        return self.experiment_dir / "last_model.pth"

    @property
    def loss_curve_path(self) -> Path:
        return self.experiment_dir / "loss_curve.png"

    @property
    def height_curve_path(self) -> Path:
        return self.experiment_dir / "height_rmse_vs_labels.png"

    @property
    def config_log_path(self) -> Path:
        return self.experiment_dir / "config.yaml"

    def make_dirs(self) -> None:
        self.viz_output_dir.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        """Snapshot the config next to the experiment outputs for reproducibility."""
        with open(self.config_log_path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)


def load_config(yaml_path: str, overrides: Optional[dict] = None) -> ExperimentConfig:
    """Load a YAML experiment file, then apply non-None CLI overrides."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return ExperimentConfig(**cfg)
