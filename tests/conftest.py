import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.utils import prepare_omegaconf

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
    cwd=True,
)
prepare_omegaconf()


def cfg_train_global(overrides=None) -> DictConfig:
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=overrides)

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(ROOT)
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.datamodule.num_workers = 0
            cfg.datamodule.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


def cfg_eval_global(overrides=None) -> DictConfig:
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(config_name="evaluate.yaml", return_hydra_config=True, overrides=overrides)

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(ROOT)
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.datamodule.num_workers = 0
            cfg.datamodule.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


def cfg_predict_global(overrides=None) -> DictConfig:
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(config_name="predict.yaml", return_hydra_config=True, overrides=overrides)

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(ROOT)
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_train(tmp_path) -> DictConfig:
    # use bert tiny for all tests
    base_model = "prajjwal1/bert-tiny"
    cfg = cfg_train_global(
        overrides=[
            f"model.model_name_or_path={base_model}",
            f"taskmodule.tokenizer_name_or_path={base_model}",
        ]
    )

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.save_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


# this is called by each test which uses `cfg_eval` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_eval(tmp_path) -> DictConfig:
    cfg = cfg_eval_global()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


# this is called by each test which uses `cfg_eval` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_predict(tmp_path) -> DictConfig:
    cfg = cfg_predict_global()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.save_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
