from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from emb2heights.config import ExperimentConfig, load_config


def minimal(**kw):
    base = dict(experiment_name="e", train_embeddings_dir="a", train_targets_dir="b")
    base.update(kw)
    return ExperimentConfig(**base)


def test_derived_paths_follow_experiment_name():
    cfg = minimal(experiment_name="my_exp", base_dir="out")
    assert cfg.experiment_dir == Path("out/my_exp")
    assert cfg.best_model_path == Path("out/my_exp/best_model.pth")
    assert cfg.viz_output_dir == Path("out/my_exp/visualizations")


def test_invalid_model_name_rejected():
    with pytest.raises(ValidationError):
        minimal(model_name="not_a_model")


def test_defaults_match_task():
    cfg = minimal()
    assert cfg.n_classes == 4
    assert cfg.height_normalization_constant == 30.0


def test_load_config_applies_only_non_none_overrides(tmp_path):
    yaml_path = tmp_path / "exp.yaml"
    yaml_path.write_text(yaml.safe_dump(dict(
        experiment_name="from_yaml", train_embeddings_dir="a",
        train_targets_dir="b", batch_size=16,
    )))
    cfg = load_config(str(yaml_path), overrides={"batch_size": 4, "epochs": None})
    assert cfg.batch_size == 4          # explicitly overridden
    assert cfg.epochs == 30             # None override ignored -> default kept
    assert cfg.experiment_name == "from_yaml"


def test_save_roundtrip(tmp_path):
    cfg = minimal(base_dir=str(tmp_path))
    cfg.make_dirs()
    cfg.save()
    reloaded = ExperimentConfig(**yaml.safe_load(cfg.config_log_path.read_text()))
    assert reloaded == cfg
