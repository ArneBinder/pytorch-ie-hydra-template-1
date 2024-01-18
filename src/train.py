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
from typing import Any, Dict, List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pie_datasets import DatasetDict
from pie_modules.models.interface import RequiresTaskmoduleConfig
from pytorch_ie.core import PyTorchIEModel, TaskModule
from pytorch_ie.models import *  # noqa: F403
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses
from pytorch_ie.taskmodules import *  # noqa: F403
from pytorch_ie.taskmodules.interface import ChangesTokenizerVocabSize
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import Logger

from src import utils
from src.datamodules import PieDataModule
from src.models import *  # noqa: F403
from src.taskmodules import *  # noqa: F403

log = utils.get_pylogger(__name__)


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


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Init pytorch-ie taskmodule
    log.info(f"Instantiating taskmodule <{cfg.taskmodule._target_}>")
    taskmodule: TaskModule = hydra.utils.instantiate(cfg.taskmodule, _convert_="partial")

    # Init pytorch-ie dataset
    log.info(f"Instantiating dataset <{cfg.dataset._target_}>")
    dataset: DatasetDict = hydra.utils.instantiate(
        cfg.dataset,
        _convert_="partial",
    )

    # auto-convert the dataset if the taskmodule specifies a document type
    dataset = taskmodule.convert_dataset(dataset)

    # Init pytorch-ie datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: PieDataModule = hydra.utils.instantiate(
        cfg.datamodule, dataset=dataset, taskmodule=taskmodule, _convert_="partial"
    )
    # Use the train dataset split to prepare the taskmodule
    taskmodule.prepare(dataset[datamodule.train_split])

    # Init the pytorch-ie model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    # get additional model arguments
    additional_model_kwargs: Dict[str, Any] = {}
    model_cls = hydra.utils.get_class(cfg.model["_target_"])
    # NOTE: MODIFY THE additional_model_kwargs IF YOUR MODEL REQUIRES ANY MORE PARAMETERS FROM THE TASKMODULE!
    # SEE EXAMPLES BELOW.
    if issubclass(model_cls, RequiresNumClasses):
        additional_model_kwargs["num_classes"] = len(taskmodule.label_to_id)
    if issubclass(model_cls, RequiresModelNameOrPath):
        if "model_name_or_path" not in cfg.model:
            raise Exception(
                f"Please specify model_name_or_path in the model config for {model_cls.__name__}."
            )
    if isinstance(taskmodule, ChangesTokenizerVocabSize):
        additional_model_kwargs["tokenizer_vocab_size"] = len(taskmodule.tokenizer)

    if issubclass(model_cls, RequiresTaskmoduleConfig):
        additional_model_kwargs["taskmodule_config"] = taskmodule.config

    # initialize the model
    model: PyTorchIEModel = hydra.utils.instantiate(
        cfg.model, _convert_="partial", **additional_model_kwargs
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_dict_entries(cfg, key="callbacks")

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_dict_entries(cfg, key="logger")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "dataset": dataset,
        "taskmodule": taskmodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(logger=logger, model=model, taskmodule=taskmodule, config=cfg)

    if cfg.model_save_dir is not None:
        log.info(f"Save taskmodule to {cfg.model_save_dir} [push_to_hub={cfg.push_to_hub}]")
        taskmodule.save_pretrained(save_directory=cfg.model_save_dir, push_to_hub=cfg.push_to_hub)
    else:
        log.warning("the taskmodule is not saved because no save_dir is specified")

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if best_ckpt_path != "":
        log.info(f"Best ckpt path: {best_ckpt_path}")
        best_checkpoint_file = os.path.basename(best_ckpt_path)
        utils.log_hyperparameters(
            logger=logger,
            best_checkpoint=best_checkpoint_file,
            checkpoint_dir=trainer.checkpoint_callback.dirpath,
        )

    if not cfg.trainer.get("fast_dev_run"):
        if cfg.model_save_dir is not None:
            if best_ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for saving...")
            else:
                model = type(model).load_from_checkpoint(best_ckpt_path)

            log.info(f"Save model to {cfg.model_save_dir} [push_to_hub={cfg.push_to_hub}]")
            model.save_pretrained(save_directory=cfg.model_save_dir, push_to_hub=cfg.push_to_hub)
        else:
            log.warning("the model is not saved because no save_dir is specified")

    if cfg.get("validate"):
        log.info("Starting validation!")
        if best_ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for validation...")
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=best_ckpt_path or None)
    elif cfg.get("train"):
        log.warning(
            "Validation after training is skipped! That means, the finally reported validation scores are "
            "the values from the *last* checkpoint, not from the *best* checkpoint (which is saved)!"
        )

    if cfg.get("test"):
        log.info("Starting testing!")
        if best_ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt_path or None)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    # add model_save_dir to the result so that it gets dumped to job_return_value.json
    # if we use hydra_callbacks.SaveJobReturnValueCallback
    if cfg.get("model_save_dir") is not None:
        metric_dict["model_save_dir"] = cfg.model_save_dir

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    if cfg.get("optimized_metric") is not None:
        metric_value = get_metric_value(
            metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
        )

        # return optimized metric
        return metric_value
    else:
        return metric_dict


if __name__ == "__main__":
    utils.replace_sys_args_with_values_from_files()
    utils.prepare_omegaconf()
    main()
