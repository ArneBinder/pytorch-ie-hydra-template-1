from os import path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.predict import predict
from tests.helpers.run_if import RunIf


@pytest.mark.slow
def test_predict_cpu(tmp_path, cfg_predict):
    """Run inference on CPU."""

    HydraConfig().set_config(cfg_predict)
    predict(cfg_predict)

    assert path.exists(path.join(cfg_predict.prediction_save_dir, "documents.jsonl"))
    assert path.exists(path.join(cfg_predict.prediction_save_dir, "config.yaml"))


def test_predict_cpu_fast_dev_run(tmp_path, cfg_predict):
    """Run inference for one batch on CPU."""
    with open_dict(cfg_predict):
        cfg_predict.pipeline.fast_dev_run = True

    HydraConfig().set_config(cfg_predict)
    predict(cfg_predict)

    assert path.exists(path.join(cfg_predict.prediction_save_dir, "documents.jsonl"))
    assert path.exists(path.join(cfg_predict.prediction_save_dir, "config.yaml"))


@RunIf(min_gpus=1)
def test_predict_gpu(tmp_path, cfg_predict):
    """Run inference on GPU."""

    with open_dict(cfg_predict):
        cfg_predict.pipeline.device = 0

    HydraConfig().set_config(cfg_predict)
    predict(cfg_predict)

    assert path.exists(path.join(cfg_predict.prediction_save_dir, "documents.jsonl"))
    assert path.exists(path.join(cfg_predict.prediction_save_dir, "config.yaml"))
