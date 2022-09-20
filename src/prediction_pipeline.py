import os.path
from typing import Callable, Dict, Sequence

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_ie.core import Document
from pytorch_ie.data import Dataset
from pytorch_ie.pipeline import Pipeline
from pytorch_lightning import seed_everything

from src import utils

log = utils.get_logger(__name__)


def predict(config: DictConfig) -> None:
    """Contains minimal example of the prediction pipeline.
    Uses a pretrained model to annotate documents from a dataset and serializes them.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert to absolute path
    absolute_path = hydra.utils.to_absolute_path(config.model_name_or_path)
    # If the converted path exists locally, use it.
    # Otherwise, model_name_or_path may point to a resource at Huggingface model hub.
    if os.path.exists(absolute_path):
        config.model_name_or_path = absolute_path

    # Init pytorch-ie dataset
    log.info(f"Instantiating dataset <{config.dataset._target_}>")
    dataset: Dict[str, Dataset] = hydra.utils.instantiate(config.dataset, _convert_="partial")

    # Init pytorch-ie pipeline
    log.info(
        f"Instantiating pipeline <{config.pipeline._target_}> from {config.model_name_or_path}"
    )
    pipeline: Pipeline = hydra.utils.instantiate(
        config.pipeline,
        pretrained_model_name_or_path=config.model_name_or_path,
        _convert_="partial",
    )

    # Per default, the model is loaded with .from_pretrained() which already loads the weights.
    # However, ckpt_path can be used to load different weights from any checkpoint.
    if config.ckpt_path is not None:
        # Convert relative ckpt path to absolute path if necessary
        config.ckpt_path = hydra.utils.to_absolute_path(config.ckpt_path)
        pipeline.model = pipeline.model.load_from_checkpoint(checkpoint_path=config.ckpt_path).to(
            pipeline.device
        )

    # select the dataset split for prediction
    dataset_predict = dataset[config.dataset_split]

    log.info("Starting inference!")
    dataset_with_predictions = pipeline(dataset_predict, inplace=False)

    # Convert to absolute path
    config.out_path = hydra.utils.to_absolute_path(config.out_path)

    # Init the serializer
    log.info(f"Instantiating serializer <{config.serializer._target_}>")
    serializer: Callable[[Sequence[Document]], None] = hydra.utils.instantiate(
        config.serializer, path=config.out_path, _convert_="partial"
    )
    # serialize the documents
    serializer(dataset_with_predictions)

    # serialize config with resolved paths
    if config.get("config_out_path"):
        config.config_out_path = hydra.utils.to_absolute_path(config.config_out_path)
        OmegaConf.save(config=config, f=config.config_out_path)
