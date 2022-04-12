from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_ie.core.pytorch_ie import PyTorchIEModel
from pytorch_ie.data.datamodules.datamodule import DataModule
from pytorch_ie.data.datasets import PIEDatasetDict
from pytorch_ie.taskmodules.taskmodule import TaskModule
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline.
    Can additionally evaluate model on a testset, using best weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = config.trainer.get("resume_from_checkpoint")
    if ckpt_path:
        config.trainer.resume_from_checkpoint = hydra.utils.to_absolute_path(ckpt_path)

    # Init pytorch-ie documents
    log.info(f"Instantiating documents <{config.documents._target_}>")
    documents: PIEDatasetDict = hydra.utils.instantiate(config.documents)

    # Init pytorch-ie taskmodule
    log.info(f"Instantiating taskmodule <{config.taskmodule._target_}>")
    taskmodule: TaskModule = hydra.utils.instantiate(config.taskmodule)

    # Init pytorch-ie datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: DataModule = hydra.utils.instantiate(
        config.datamodule, dataset=documents, taskmodule=taskmodule
    )
    # This calls taskmodule.prepare() on the train split.
    datamodule.setup(stage="fit")

    # Init taskmodule-model-bridge
    additional_model_kwargs: Dict[str, Any] = hydra.utils.instantiate(
        config.bridge, taskmodule=taskmodule
    )

    # Init pytorch-ie model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: PyTorchIEModel = hydra.utils.instantiate(config.model, **additional_model_kwargs)

    # Init lightning callbacks
    callbacks = utils.instantiate_dict_entries(config, "callbacks")

    # Init lightning loggers
    logger = utils.instantiate_dict_entries(config, "logger")

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    save_dir = hydra.utils.to_absolute_path(config["save_dir"])

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        taskmodule=taskmodule,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    log.info(f"Save taskmodule to {save_dir} [push_to_hub={config.push_to_hub}]")
    taskmodule.save_pretrained(save_directory=save_dir, push_to_hub=config.push_to_hub)

    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test"):
        ckpt_path = "best"
        if not config.get("train") or config.trainer.get("fast_dev_run"):
            ckpt_path = None
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        taskmodule=taskmodule,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")
        model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        log.info(f"Save best model to {save_dir} [push_to_hub={config.push_to_hub}]")
        model.save_pretrained(save_directory=save_dir, push_to_hub=config.push_to_hub)

    # Return metric score for hyperparameter optimization
    return score
