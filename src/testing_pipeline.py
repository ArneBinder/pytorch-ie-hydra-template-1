import hydra
from omegaconf import DictConfig
from pytorch_ie.core.pytorch_ie import PyTorchIEModel
from pytorch_ie.data.datamodules.datamodule import DataModule
from pytorch_ie.data.datasets import PIEDatasetDict
from pytorch_ie.taskmodules.taskmodule import TaskModule
from pytorch_lightning import Trainer, seed_everything

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

    # Convert to absolute path if necessary
    config.pretrained_model_name_or_path = hydra.utils.to_absolute_path(
        config.pretrained_model_name_or_path
    )

    # Per default, the model is loaded with .from_pretrained() which already loads the weights.
    # However, ckpt_path can be used to load different weights from any checkpoint.
    if config.ckpt_path is not None:
        # Convert relative ckpt path to absolute path if necessary
        config.ckpt_path = hydra.utils.to_absolute_path(config.ckpt_path)

    # Init pytorch-ie dataset
    log.info(f"Instantiating dataset <{config.dataset._target_}>")
    dataset: PIEDatasetDict = hydra.utils.instantiate(config.dataset)

    # Init pytorch-ie taskmodule
    log.info(f"Instantiating taskmodule <{config.taskmodule._target_}>")
    taskmodule: TaskModule = hydra.utils.instantiate(
        config.taskmodule, pretrained_model_name_or_path=config.pretrained_model_name_or_path
    )

    # Init pytorch-ie datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: DataModule = hydra.utils.instantiate(
        config.datamodule, dataset=dataset, taskmodule=taskmodule
    )
    datamodule.setup(stage="test")

    # Init pytorch-ie model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: PyTorchIEModel = hydra.utils.instantiate(
        config.model, pretrained_model_name_or_path=config.pretrained_model_name_or_path
    )

    # Init lightning loggers
    logger = utils.instantiate_dict_entries(config, "logger")

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    # Log hyperparameters
    trainer.logger.log_hyperparams(
        {
            "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
            "ckpt_path": config.ckpt_path,
            "dataset": config.dataset,
            # Note: we log the config from the instantiated objects to log the real hparams,
            # the hydra configs just contain the path and the object types
            "model": model._config,
            "taskmodule": taskmodule._config,
        }
    )

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)
