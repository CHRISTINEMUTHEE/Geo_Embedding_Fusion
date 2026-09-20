import math

import pytest
import torch

from emb2heights.losses import build_loss
from emb2heights.models import build_model
from emb2heights.trainers import (
    _run_epoch,
    binary_iou_from_channel,
    build_optimizer,
    build_scheduler,
    evaluate_metrics,
    masked_rmse,
    train,
)


# ---------------- metrics: hand-computed values ----------------
def test_binary_iou_hand_computed():
    ## pred hits {(0,0),(1,1)}, target hits {(0,0),(0,1)} -> intersection 1, union 3
    pred = torch.tensor([[[0.5, 0.0], [0.0, 0.5]]])
    target = torch.tensor([[[0.5, 0.5], [0.0, 0.0]]])
    iou = binary_iou_from_channel(pred, target, threshold=0.1)
    assert iou == pytest.approx(1 / 3, abs=1e-4)


def test_binary_iou_empty_union_is_nan():
    zeros = torch.zeros(1, 2, 2)
    assert math.isnan(binary_iou_from_channel(zeros, zeros).item())


def test_masked_rmse_hand_computed():
    pred = torch.tensor([[[3.0, 4.0]]])
    true = torch.tensor([[[1.0, 2.0]]])
    mask = torch.ones_like(pred, dtype=torch.bool)
    assert masked_rmse(pred, true, mask).item() == pytest.approx(2.0)


def test_masked_rmse_empty_mask_is_nan():
    pred = torch.zeros(1, 2, 2)
    mask = torch.zeros_like(pred, dtype=torch.bool)
    assert math.isnan(masked_rmse(pred, pred, mask).item())


# ---------------- builders ----------------
def test_build_optimizer_and_scheduler_variants(config):
    model = build_model(config, n_channels=4)
    for opt_name, opt_cls in [("adam", torch.optim.Adam),
                              ("adamw", torch.optim.AdamW),
                              ("sgd", torch.optim.SGD)]:
        cfg = config.model_copy(update={"optimizer": opt_name})
        assert isinstance(build_optimizer(cfg, model), opt_cls)

    opt = build_optimizer(config, model)
    for sched_name in ["plateau", "cosine", "step"]:
        cfg = config.model_copy(update={"scheduler": sched_name})
        assert build_scheduler(cfg, opt) is not None
    assert build_scheduler(config.model_copy(update={"scheduler": "none"}), opt) is None


def test_unknown_optimizer_raises(config):
    model = build_model(config, n_channels=4)
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_optimizer(config.model_copy(update={"optimizer": "nope"}), model)


# ---------------- training loop ----------------
def test_run_epoch_trains_weights(config):
    from emb2heights.datasets import build_dataloaders
    train_loader, _ = build_dataloaders(config)
    model = build_model(config, n_channels=8)
    criterion = build_loss(config)
    optimizer = build_optimizer(config, model)

    before = [p.detach().clone() for p in model.parameters()]
    loss = _run_epoch(model, train_loader, criterion, torch.device("cpu"), optimizer)
    assert math.isfinite(loss)
    changed = any(not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters()))
    assert changed


def test_run_epoch_eval_does_not_touch_weights(config):
    from emb2heights.datasets import build_dataloaders
    _, val_loader = build_dataloaders(config)
    model = build_model(config, n_channels=8)
    before = [p.detach().clone() for p in model.parameters()]
    loss = _run_epoch(model, val_loader, build_loss(config), torch.device("cpu"))
    assert math.isfinite(loss)
    assert all(torch.equal(b, p.detach()) for b, p in zip(before, model.parameters()))


def test_evaluate_metrics_returns_all_keys(config):
    from emb2heights.datasets import build_dataloaders
    _, val_loader = build_dataloaders(config)
    model = build_model(config, n_channels=8)
    metrics = evaluate_metrics(model, val_loader, torch.device("cpu"),
                               config.height_normalization_constant)
    assert set(metrics) == {"iou_building", "iou_vegetation", "iou_water",
                            "mae_height", "rmse_height",
                            "rmse_building", "rmse_vegetation"}


# ---------------- end to end ----------------
def test_train_end_to_end_produces_artifacts(config):
    model, history = train(config)
    assert len(history["train_losses"]) == config.epochs
    assert math.isfinite(history["best_val_loss"])
    for artifact in [config.best_model_path, config.last_model_path,
                     config.loss_curve_path, config.height_curve_path,
                     config.config_log_path]:
        assert artifact.exists(), f"missing artifact: {artifact}"
    ## weights reload cleanly into a fresh model of the same shape
    fresh = build_model(config, n_channels=8)
    fresh.load_state_dict(torch.load(config.best_model_path, weights_only=True))
