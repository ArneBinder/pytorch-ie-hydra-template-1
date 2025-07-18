import os

import pytest
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from tests.conftest import cfg_train_global

# strip extension to have nicer logging output, e.g. tests/test_experiments.py::test_experiments[conll2003]
ext = ".yaml"
AVAILABLE_EXPERIMENTS = [
    exp_yaml.replace(ext, "")
    for exp_yaml in os.listdir("configs/experiment")
    if exp_yaml.endswith(ext) and not exp_yaml.startswith("_")
]

EXPERIMENTS = AVAILABLE_EXPERIMENTS


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function", params=EXPERIMENTS)
def cfg_experiment(tmp_path, request) -> DictConfig:
    cfg = cfg_train_global(overrides=[f"experiment={request.param}"]).copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.save_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.mark.slow
@pytest.mark.veryslow
def test_experiment(cfg_experiment):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_experiment)
    with open_dict(cfg_experiment):
        cfg_experiment.trainer.fast_dev_run = True
        cfg_experiment.trainer.accelerator = "cpu"
    train(cfg_experiment)
