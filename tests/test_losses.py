import torch

from emb2heights.config import ExperimentConfig
from emb2heights.losses import build_loss


def test_build_loss_is_mae():
    cfg = ExperimentConfig(experiment_name="e", train_embeddings_dir="a",
                           train_targets_dir="b")
    loss = build_loss(cfg)
    pred = torch.tensor([1.0, 3.0])
    true = torch.tensor([0.0, 1.0])
    assert loss(pred, true).item() == 1.5  # mean(|1|, |2|)
