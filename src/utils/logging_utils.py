import logging
from importlib.util import find_spec
from typing import List, Optional, Union

from omegaconf import DictConfig, OmegaConf
from pytorch_ie import PyTorchIEModel, TaskModule
from pytorch_ie.models.interface import RequiresTaskmoduleConfig
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_pylogger(__name__)


@rank_zero_only
def log_hyperparameters(
    logger: Optional[List[Logger]] = None,
    config: Optional[Union[dict, DictConfig]] = None,
    model: Optional[PyTorchIEModel] = None,
    taskmodule: Optional[TaskModule] = None,
    key_prefix: str = "_",
    **kwargs,
) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additional saves:
    - Number of model parameters
    """

    hparams = {}

    if not logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # this is just for backwards compatibility: usually, the taskmodule_config should be passed to
    # the model and, thus, be logged there automatically
    if model is not None and not isinstance(model, RequiresTaskmoduleConfig):
        if taskmodule is None:
            raise ValueError(
                "If model is not an instance of RequiresTaskmoduleConfig, taskmodule must be passed!"
            )
        # here we use the taskmodule/model config how it is after preparation/initialization
        hparams["taskmodule_config"] = taskmodule.config

    if model is not None:
        # save number of model parameters
        hparams[f"{key_prefix}num_params/total"] = sum(p.numel() for p in model.parameters())
        hparams[f"{key_prefix}num_params/trainable"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        hparams[f"{key_prefix}num_params/non_trainable"] = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )

    if config is not None:
        hparams[f"{key_prefix}config"] = (
            OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
        )

    # add additional hparams
    for k, v in kwargs.items():
        hparams[f"{key_prefix}{k}"] = v

    # send hparams to all loggers
    for current_logger in logger:
        current_logger.log_hyperparams(hparams)


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()
