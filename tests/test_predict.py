import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.predict import predict
from tests.helpers.run_if import RunIf


@pytest.mark.slow
def test_predict(tmp_path, cfg_predict):
    """Run inference on cpu."""

    HydraConfig().set_config(cfg_predict)
    predict(cfg_predict)

    assert os.path.exists(f"{cfg_predict.out_path}.jsonl")
    assert os.path.exists(f"{cfg_predict.out_path}.config.yaml")


def test_predict_fast_dev_run(tmp_path, cfg_predict):
    """Run inference for one batch on cpu."""
    with open_dict(cfg_predict):
        cfg_predict.pipeline.fast_dev_run = True

    HydraConfig().set_config(cfg_predict)
    predict(cfg_predict)

    assert os.path.exists(f"{cfg_predict.out_path}.jsonl")
    assert os.path.exists(f"{cfg_predict.out_path}.config.yaml")


@RunIf(min_gpus=1)
def test_predict_gpu(tmp_path, cfg_predict):
    """Run for 1 train, val and test step on GPU."""

    with open_dict(cfg_predict):
        cfg_predict.pipeline.device = 0

    HydraConfig().set_config(cfg_predict)
    predict(cfg_predict)

    assert os.path.exists(f"{cfg_predict.out_path}.jsonl")
    assert os.path.exists(f"{cfg_predict.out_path}.config.yaml")
