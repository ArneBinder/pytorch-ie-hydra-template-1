import os
from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_ie.core.pytorch_ie import PyTorchIEModel
from pytorch_ie.data.datamodules.datamodule import DataModule
from pytorch_ie.data.datasets import PIEDatasetDict
from pytorch_ie.taskmodules.taskmodule import TaskModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_logger(__name__)


def test(config: DictConfig) -> None:
    """Contains minimal example of the testing pipeline.
    Evaluates given checkpoint on a testset.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Per default, the model is loaded with .from_pretrained() which already loads the weights.
    # However, ckpt_path can be used to load different weights from any checkpoint.
    if config.ckpt_path is not None:
        # Convert relative ckpt path to absolute path if necessary
        if not os.path.isabs(config.ckpt_path):
            config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.ckpt_path)

    # Init pytorch-ie dataset
    log.info(f"Instantiating dataset <{config.dataset._target_}>")
    dataset: PIEDatasetDict = hydra.utils.instantiate(config.dataset)

    # Init pytorch-ie taskmodule
    log.info(f"Instantiating taskmodule <{config.taskmodule._target_}>")
    taskmodule: TaskModule = hydra.utils.instantiate(config.taskmodule)

    # Init pytorch-ie datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: DataModule = hydra.utils.instantiate(
        config.datamodule, dataset=dataset, taskmodule=taskmodule
    )
    datamodule.setup(stage="test")

    # Init pytorch-ie model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: PyTorchIEModel = hydra.utils.instantiate(config.model)

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    # Log hyperparameters
    trainer.logger.log_hyperparams({"ckpt_path": config.ckpt_path, "model": model._config, "taskmodule": taskmodule._config, "dataset": config.datase})

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)
