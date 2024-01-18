import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pie_datasets import DatasetDict
from pytorch_ie.core import PyTorchIEModel, TaskModule
from pytorch_lightning import Trainer

from src import utils
from src.datamodules import PieDataModule
from src.models import *  # noqa: F403
from src.taskmodules import *  # noqa: F403

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Init pytorch-ie dataset
    log.info(f"Instantiating dataset <{cfg.dataset._target_}>")
    dataset: DatasetDict = hydra.utils.instantiate(cfg.dataset, _convert_="partial")

    # Init pytorch-ie taskmodule
    log.info(f"Instantiating taskmodule <{cfg.taskmodule._target_}>")
    taskmodule: TaskModule = hydra.utils.instantiate(cfg.taskmodule, _convert_="partial")

    # auto-convert the dataset if the taskmodule specifies a document type
    dataset = taskmodule.convert_dataset(dataset)

    # Init pytorch-ie datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: PieDataModule = hydra.utils.instantiate(
        cfg.datamodule, dataset=dataset, taskmodule=taskmodule, _convert_="partial"
    )

    # Init pytorch-ie model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: PyTorchIEModel = hydra.utils.instantiate(cfg.model, _convert_="partial")

    # Init lightning loggers
    logger = utils.instantiate_dict_entries(cfg, "logger")

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, _convert_="partial")

    object_dict = {
        "cfg": cfg,
        "taskmodule": taskmodule,
        "dataset": dataset,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(logger=logger, model=model, taskmodule=taskmodule, config=cfg)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="evaluate.yaml")
def main(cfg: DictConfig) -> None:
    metric_dict, _ = evaluate(cfg)

    return metric_dict


if __name__ == "__main__":
    utils.replace_sys_args_with_values_from_files()
    utils.prepare_omegaconf()
    main()
