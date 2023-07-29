import logging
from importlib.util import find_spec

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
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additional saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    taskmodule = object_dict["taskmodule"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # choose which parts of hydra config will be saved to loggers
    # here we use the taskmodule/model config how it is after preparation/initialization
    hparams["taskmodule"] = taskmodule._config()
    hparams["model"] = model._config()

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["dataset"] = cfg["dataset"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["pipeline_type"] = cfg.get("pipeline_type")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    hparams["monitor_metric"] = cfg.get("monitor_metric")
    hparams["monitor_mode"] = cfg.get("monitor_mode")

    hparams["model_save_dir"] = cfg.get("model_save_dir")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()
