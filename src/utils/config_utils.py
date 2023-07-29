from copy import copy
from typing import Any, List, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.utils.logging_utils import get_pylogger

logger = get_pylogger(__name__)


def execute_pipeline(
    input: Any,
    setup: Optional[Any] = None,
    **processors,
) -> Any:
    if setup is not None and callable(setup):
        setup()
    result = input
    for processor_name, processor_config in processors.items():
        if not isinstance(processor_config, dict) or "_processor_" not in processor_config:
            continue
        logger.info(f"call processor: {processor_name}")
        config = copy(processor_config)
        if not config.pop("_enabled_", True):
            logger.warning(f"skip processor because it is disabled: {processor_name}")
            continue
        # rename key "_processor_" to "_target_"
        if "_target_" in config:
            raise ValueError(
                f"processor {processor_name} has a key '_target_', which is not allowed"
            )
        config["_target_"] = config.pop("_processor_")
        # IMPORTANT: We pass result as the first argument after the config in contrast to adding it to the config.
        # By doing so, we prevent that it gets converted into a OmegaConf object which would be converted back to
        # a simple dict breaking all the DatasetDict methods
        tmp_result = instantiate(config, result, _convert_="partial")
        if tmp_result is not None:
            result = tmp_result
        else:
            logger.warning(f'processor "{processor_name}" did not return a result')
    return result


def instantiate_dict_entries(
    config: DictConfig, key: str, entry_description: Optional[str] = None
) -> List:
    entries: List = []
    key_config = config.get(key)

    if not key_config:
        logger.warning(f"{key} config is empty.")
        return entries

    if not isinstance(key_config, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, entry_conf in key_config.items():
        if isinstance(entry_conf, DictConfig) and "_target_" in entry_conf:
            logger.info(f"Instantiating {entry_description or key} <{entry_conf._target_}>")
            entries.append(instantiate(entry_conf, _convert_="partial"))

    return entries


def prepare_omegaconf():
    # register replace resolver (used to replace "/" with "-" in names to use them as e.g. wandb project names)
    if not OmegaConf.has_resolver("replace"):
        OmegaConf.register_new_resolver("replace", lambda s, x, y: s.replace(x, y))
    else:
        logger.warning("OmegaConf resolver 'replace' is already registered")
