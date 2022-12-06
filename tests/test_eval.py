from os import path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.evaluate import evaluate
from src.train import train
from tests.helpers.run_if import RunIf


@pytest.mark.slow
def test_train_eval(tmp_path, cfg_train, cfg_eval):
    """Train for 1 epoch with `train.py` and evaluate with `eval.py`"""
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True
        cfg_train.trainer.limit_train_batches = 10

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert path.exists(cfg_train.model_save_dir)

    with open_dict(cfg_eval):
        cfg_eval.model_name_or_path = cfg_train.model_save_dir

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    assert test_metric_dict["test/f1"] > 0.0
    assert abs(train_metric_dict["test/f1"].item() - test_metric_dict["test/f1"].item()) < 0.00001
