import os.path
from typing import Dict

import hydra
from omegaconf import DictConfig
from pytorch_ie.core import PyTorchIEModel, TaskModule
from pytorch_ie.data import Dataset
from pytorch_ie.data.datamodules.datamodule import DataModule
from pytorch_lightning import Trainer, seed_everything

from src import utils

log = utils.get_logger(__name__)


def evaluate(config: DictConfig) -> None:
    """
    Contains minimal example of the evaluation pipeline.
    Evaluates given checkpoint on a testset.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init pytorch-ie dataset
    log.info(f"Instantiating dataset <{config.dataset._target_}>")
    dataset: Dict[str, Dataset] = hydra.utils.instantiate(config.dataset, _convert_="partial")

    # Init pytorch-ie taskmodule
    log.info(f"Instantiating taskmodule <{config.taskmodule._target_}>")
    taskmodule: TaskModule = hydra.utils.instantiate(config.taskmodule, _convert_="partial")

    # Init pytorch-ie datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: DataModule = hydra.utils.instantiate(
        config.datamodule, dataset=dataset, taskmodule=taskmodule, _convert_="partial"
    )
    datamodule.setup(stage="test")

    # Init pytorch-ie model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: PyTorchIEModel = hydra.utils.instantiate(config.model, _convert_="partial")

    # Init lightning loggers
    logger = utils.instantiate_dict_entries(config, "logger")

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger, _convert_="partial")

    # Log hyperparameters
    if trainer.logger:
        trainer.logger.log_hyperparams(
            {
                "model_name_or_path": config.model_name_or_path,
                "ckpt_path": config.ckpt_path,
                "dataset": config.dataset,
                # Note: we log the config from the instantiated objects to log the real hparams,
                # the hydra configs just contain the path and the object types
                "model": model._config,
                "taskmodule": taskmodule._config,
            }
        )
    else:
        log.warning("can not log hyperparameters because no logger is configured")

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)
