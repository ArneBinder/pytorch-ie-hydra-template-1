from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig
from pytorch_ie.core import PyTorchIEModel, TaskModule
from pytorch_ie.data import Dataset
from pytorch_ie.data.datamodules.datamodule import DataModule
from pytorch_lightning import Trainer, seed_everything

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
    # This calls taskmodule.prepare() on the train split.
    datamodule.setup(stage="fit")

    # Init pytorch-ie model
    log.info(f"Instantiating model <{config.model._target_}>")
    # NOTE: THE FOLLOWING LINE MAY NEED ADAPTATION WHEN YOU DEFINE YOUR OWN MODELS OR TASKMODULES!
    additional_model_kwargs: Dict[str, Any] = dict(num_classes=len(taskmodule.label_to_id))
    model: PyTorchIEModel = hydra.utils.instantiate(
        config.model, _convert_="partial", **additional_model_kwargs
    )

    # Init lightning callbacks
    callbacks = utils.instantiate_dict_entries(config, "callbacks")

    # Init lightning loggers
    logger = utils.instantiate_dict_entries(config, "logger")

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    if config.save_dir is not None:
        config.save_dir = hydra.utils.to_absolute_path(config.save_dir)

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

    if config.save_dir is not None:
        log.info(f"Save taskmodule to {config.save_dir} [push_to_hub={config.push_to_hub}]")
        taskmodule.save_pretrained(save_directory=config.save_dir, push_to_hub=config.push_to_hub)
    else:
        log.warning("the taskmodule is not saved because no save_dir is specified")

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
        if config.save_dir is not None:
            log.info(f"Save best model to {config.save_dir} [push_to_hub={config.push_to_hub}]")
            model.save_pretrained(save_directory=config.save_dir, push_to_hub=config.push_to_hub)
        else:
            log.warning("the model is not saved because no save_dir is specified")

    # Return metric score for hyperparameter optimization
    return score
