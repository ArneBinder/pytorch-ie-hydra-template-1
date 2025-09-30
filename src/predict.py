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

import os
from typing import Any, Dict, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pie_core import AnnotationPipeline
from pie_datasets import DatasetDict
from pytorch_ie import PyTorchIEPipeline
from pytorch_ie.models import *  # noqa: F403
from pytorch_ie.taskmodules import *  # noqa: F403

from src import utils
from src.models import *  # noqa: F403
from src.serializer.interface import DocumentSerializer
from src.taskmodules import *  # noqa: F403

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
    # The pipeline, and therefore the inference step, is optional to allow for easy testing
    # of the dataset creation and processing.
    pipeline: Optional[AnnotationPipeline] = None
    if cfg.get("pipeline") and cfg.pipeline.get("_target_"):
        log.info(f"Instantiating pipeline <{cfg.pipeline._target_}> from {cfg.model_name_or_path}")
        pipeline = hydra.utils.instantiate(cfg.pipeline, _convert_="partial")

        # Per default, the model is loaded with .from_pretrained() which already loads the weights.
        # However, ckpt_path can be used to load different weights from any checkpoint.
        if cfg.ckpt_path is not None:
            log.info(f"Loading model weights from checkpoint: {cfg.ckpt_path}")
            if not isinstance(pipeline, PyTorchIEPipeline):
                raise ValueError(
                    "The pipeline has to be of type PyTorchIEPipeline to load a checkpoint."
                )
            pipeline.model = (
                type(pipeline.model)
                .load_from_checkpoint(checkpoint_path=cfg.ckpt_path)
                .to(pipeline.device)
                .to(dtype=pipeline.model.dtype)
            )

        # auto-convert the dataset if the taskmodule specifies a document type
        dataset = dataset.to_document_type(pipeline.taskmodule, downcast=False)

    # Init the serializer
    serializer: Optional[DocumentSerializer] = None
    if cfg.get("serializer") and cfg.serializer.get("_target_"):
        log.info(f"Instantiating serializer <{cfg.serializer._target_}>")
        serializer = hydra.utils.instantiate(
            cfg.serializer, split=cfg.dataset_split, _convert_="partial"
        )

    # select the dataset split for prediction
    dataset_predict = dataset[cfg.dataset_split]

    object_dict = {
        "cfg": cfg,
        "dataset": dataset,
        "pipeline": pipeline,
        "serializer": serializer,
    }
    # predict and serialize
    result: Dict[str, Any] = utils.predict_and_serialize(
        pipeline=pipeline,
        serializer=serializer,
        dataset=dataset_predict,
        document_batch_size=cfg.get("document_batch_size", None),
    )

    # serialize config with resolved paths
    if cfg.get("config_out_path"):
        config_out_dir = os.path.dirname(cfg.config_out_path)
        os.makedirs(config_out_dir, exist_ok=True)
        OmegaConf.save(config=cfg, f=cfg.config_out_path)
        result["config"] = cfg.config_out_path

    return result, object_dict


@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    result_dict, _ = predict(cfg)
    return result_dict


if __name__ == "__main__":
    utils.replace_sys_args_with_values_from_files()
    utils.prepare_omegaconf()
    main()
