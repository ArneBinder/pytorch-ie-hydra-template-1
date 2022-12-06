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

from typing import Callable, Sequence, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_ie import DatasetDict, Pipeline
from pytorch_ie.core import Document

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def predict(cfg: DictConfig) -> Tuple[dict, dict]:
    """Contains minimal example of the prediction pipeline. Uses a pretrained model to annotate
    documents from a dataset and serializes them.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Init pytorch-ie dataset
    log.info(f"Instantiating dataset <{cfg.dataset._target_}>")
    dataset: DatasetDict = hydra.utils.instantiate(cfg.dataset, _convert_="partial")

    # Init pytorch-ie pipeline
    log.info(f"Instantiating pipeline <{cfg.pipeline._target_}> from {cfg.model_name_or_path}")
    pipeline: Pipeline = hydra.utils.instantiate(cfg.pipeline, _convert_="partial")

    # Per default, the model is loaded with .from_pretrained() which already loads the weights.
    # However, ckpt_path can be used to load different weights from any checkpoint.
    if cfg.ckpt_path is not None:
        pipeline.model = pipeline.model.load_from_checkpoint(checkpoint_path=cfg.ckpt_path).to(
            pipeline.device
        )

    # select the dataset split for prediction
    dataset_predict = dataset[cfg.dataset_split]

    log.info("Starting inference!")
    dataset_with_predictions = pipeline(dataset_predict, inplace=False)

    # Init the serializer
    log.info(f"Instantiating serializer <{cfg.serializer._target_}>")
    serializer: Callable[[Sequence[Document]], None] = hydra.utils.instantiate(
        cfg.serializer, _convert_="partial"
    )
    # serialize the documents
    serializer(dataset_with_predictions)

    # serialize config with resolved paths
    if cfg.get("config_out_path"):
        OmegaConf.save(config=cfg, f=cfg.config_out_path)

    return {}, {}


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    predict(cfg)


if __name__ == "__main__":
    utils.prepare_omegaconf()
    main()