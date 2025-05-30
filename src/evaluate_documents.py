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

from typing import Any, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pie_datasets import DatasetDict
from pytorch_ie.core import DocumentMetric

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate_documents(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates serialized PIE documents.

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
    log.info(f"Instantiating metric <{cfg.metric._target_}>")
    metric: DocumentMetric = hydra.utils.instantiate(cfg.metric, _convert_="partial")

    # auto-convert the dataset if the metric specifies a document type
    dataset = dataset.to_document_type(metric, downcast=False)

    # Init lightning loggers
    loggers = utils.instantiate_dict_entries(cfg, "logger")

    object_dict = {
        "cfg": cfg,
        "dataset": dataset,
        "metric": metric,
        "logger": loggers,
    }

    if loggers:
        log.info("Logging hyperparameters!")
        # send hparams to all loggers
        for logger in loggers:
            logger.log_hyperparams(cfg)

    splits = cfg.get("splits", None)
    if splits is None:
        documents = dataset
    else:
        documents = type(dataset)({k: v for k, v in dataset.items() if k in splits})

    metric_dict = metric(documents)

    return metric_dict, object_dict


@hydra.main(
    version_base="1.2", config_path=str(root / "configs"), config_name="evaluate_documents.yaml"
)
def main(cfg: DictConfig) -> Any:
    metric_dict, _ = evaluate_documents(cfg)
    return metric_dict


if __name__ == "__main__":
    utils.replace_sys_args_with_values_from_files()
    utils.prepare_omegaconf()
    main()
