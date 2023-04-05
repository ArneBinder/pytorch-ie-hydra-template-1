import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # apply extra utilities
        extras(cfg)

        # execute the task
        start_time = time.time()
        try:
            task_result = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.pipeline_type}' execution time: {time.time() - start_time} (s)"
            save_file(path, content)  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return task_result

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def instantiate_dict_entries(
    config: DictConfig, key: str, entry_description: Optional[str] = None
) -> List:
    entries: List = []
    key_config = config.get(key)

    if not key_config:
        log.warning(f"{key} config is empty.")
        return entries

    if not isinstance(key_config, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, entry_conf in key_config.items():
        if isinstance(entry_conf, DictConfig) and "_target_" in entry_conf:
            log.info(f"Instantiating {entry_description or key} <{entry_conf._target_}>")
            entries.append(hydra.utils.instantiate(entry_conf, _convert_="partial"))

    return entries


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


def get_metric_value(metric_dict: dict, metric_name: str) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


def prepare_omegaconf():
    # register replace resolver (used to replace "/" with "-" in names to use them as e.g. wandb project names)
    if not OmegaConf.has_resolver("replace"):
        OmegaConf.register_new_resolver("replace", lambda s, x, y: s.replace(x, y))
    else:
        log.warning("OmegaConf resolver 'replace' is already registered")
