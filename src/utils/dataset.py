import json
from copy import copy
from typing import Callable, Dict, Optional, Type, TypeVar, Union

from hydra._internal.instantiate._instantiate2 import _resolve_target, instantiate
from pytorch_ie import Dataset
from pytorch_ie.core import Document
from typing_extensions import SupportsIndex

import datasets
from src.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)


def process_dataset(input: datasets.DatasetDict, **processors) -> datasets.DatasetDict:
    result = input
    for processor_name, processor_config in processors.items():
        logger.info(f"process dataset: {processor_name}")
        config = copy(processor_config)
        # rename key "_processor_" to "_target_"
        config["_target_"] = config.pop("_processor_")
        config["dataset"] = result
        tmp_result = instantiate(config=config, _convert_="partial")
        if tmp_result is not None:
            result = tmp_result
        else:
            logger.warning(f"dataset processor {processor_name} did not return a result")
    return result


D = TypeVar("D", bound=Document)


def convert_documents(
    dataset: datasets.DatasetDict,
    function: Optional[Union[Callable, str]] = None,
    result_type: Optional[Union[str, Type[D]]] = None,
    **kwargs,
) -> datasets.DatasetDict:
    if function is not None:
        func = _resolve_target(function, full_key="")
    else:

        def identity(x):
            return x

        func = identity
    map_kwargs = dict(function=func, fn_kwargs=kwargs)
    if result_type is not None:
        result_type = _resolve_target(result_type, full_key="")
        map_kwargs["result_document_type"] = result_type
    dataset = type(dataset)({k: v.map(**map_kwargs) for k, v in dataset.items()})

    return dataset


def select(
    dataset: datasets.DatasetDict,
    split: str,
    start: Optional[SupportsIndex] = None,
    stop: Optional[SupportsIndex] = None,
    step: Optional[SupportsIndex] = None,
    **kwargs,
) -> datasets.DatasetDict:
    if stop is not None:
        range_args = [stop]
        if start is not None:
            range_args = [start] + range_args
        if step is not None:
            range_args = range_args + [step]
        kwargs["indices"] = range(*range_args)
    pie_split = dataset[split]
    if "indices" in kwargs:
        dataset[split] = Dataset.from_hf_dataset(
            dataset=pie_split.select(**kwargs), document_type=dataset[split].document_type
        )
    else:
        if len(kwargs) > 0:
            logger.warning(
                f"arguments for dataset.select() available, but they do not contain 'indices' which is required, "
                f"so we do not call select. provided arguments: \n{json.dumps(kwargs, indent=2)}"
            )
    return dataset


def rename_splits(
    dataset: datasets.DatasetDict,
    mapping: Optional[Dict[str, str]] = None,
) -> datasets.DatasetDict:
    if mapping is None:
        mapping = {}
    result = datasets.DatasetDict(
        {mapping.get(name, name): data for name, data in dataset.items()}
    )
    return result


def add_test_split(
    dataset: datasets.DatasetDict,
    source_split: str = "train",
    rename_result_splits: Optional[Dict[str, str]] = None,
    **kwargs,
) -> datasets.DatasetDict:
    split_result_hf = dataset[source_split].train_test_split(**kwargs)
    split_result = datasets.DatasetDict(
        {
            name: Dataset.from_hf_dataset(ds, document_type=dataset[source_split].document_type)
            for name, ds in split_result_hf.items()
        }
    )
    split_result = rename_splits(dataset=split_result, mapping=rename_result_splits)
    res = copy(dataset)
    res.update(split_result)
    return res
