import pytest
import torch

from emb2heights.config import ExperimentConfig
from emb2heights.models import LightUNet, build_model


def make_config(**kw):
    return ExperimentConfig(experiment_name="e", train_embeddings_dir="a",
                            train_targets_dir="b", **kw)


def test_build_model_returns_lightunet():
    model = build_model(make_config(), n_channels=8)
    assert isinstance(model, LightUNet)
    assert model.n_channels == 8
    assert model.n_classes == 4


def test_forward_output_shape_matches_input_resolution():
    model = build_model(make_config(), n_channels=8)
    out = model(torch.randn(2, 8, 32, 32))
    assert out.shape == (2, 4, 32, 32)


def test_forward_works_at_realistic_channel_counts():
    ## AlphaEarth=64; guards against hardcoded channel assumptions
    model = build_model(make_config(), n_channels=64)
    out = model(torch.randn(1, 64, 32, 32))
    assert out.shape == (1, 4, 32, 32)


def test_gradients_flow():
    model = build_model(make_config(), n_channels=4)
    out = model(torch.randn(1, 4, 32, 32))
    out.mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
