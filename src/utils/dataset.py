import json
from copy import copy
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import datasets
from hydra._internal.instantiate._instantiate2 import _resolve_target, instantiate
from pytorch_ie import Dataset, IterableDataset
from pytorch_ie.core import Document
from typing_extensions import SupportsIndex

from src.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)


def get_split_type(
    dataset_split: Union[datasets.Dataset, datasets.IterableDataset]
) -> Union[Type[Dataset], Type[IterableDataset]]:
    if isinstance(dataset_split, datasets.Dataset):
        return Dataset
    elif isinstance(dataset_split, datasets.IterableDataset):
        return IterableDataset
    else:
        raise ValueError(
            f"dataset_split must be of type Dataset or IterableDataset, but is {type(dataset_split)}"
        )


def from_hf_dataset(
    hf_dataset: datasets.DatasetDict, document_type: Union[str, Type[Document]]
) -> datasets.DatasetDict:
    if isinstance(document_type, str):
        document_type = _resolve_target(document_type, full_key="")
    res = type(hf_dataset)(
        {
            k: get_split_type(v).from_hf_dataset(v, document_type=document_type)
            for k, v in hf_dataset.items()
        }
    )
    return res


def process_dataset(
    input: datasets.DatasetDict, setup: Optional[Any] = None, **processors
) -> datasets.DatasetDict:
    if setup is not None and callable(setup):
        setup()
    result = input
    for processor_name, processor_config in processors.items():
        if not isinstance(processor_config, dict) or "_processor_" not in processor_config:
            continue
        logger.info(f"process dataset: {processor_name}")
        config = copy(processor_config)
        if not config.pop("_enabled_", True):
            logger.warning(f"skip processor because it is disabled: {processor_name}")
            continue
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
    keep_other_splits: bool = True,
) -> datasets.DatasetDict:
    if mapping is None:
        mapping = {}
    result = datasets.DatasetDict(
        {
            mapping.get(name, name): data
            for name, data in dataset.items()
            if name in mapping or keep_other_splits
        }
    )
    return result


def add_test_split(
    dataset: datasets.DatasetDict,
    source_split: str = "train",
    target_split: str = "test",
    **kwargs,
) -> datasets.DatasetDict:
    split_result_hf = dataset[source_split].train_test_split(**kwargs)
    split_result = datasets.DatasetDict(
        {
            name: Dataset.from_hf_dataset(ds, document_type=dataset[source_split].document_type)
            for name, ds in split_result_hf.items()
        }
    )
    res = copy(dataset)
    res[source_split] = split_result["train"]
    res[target_split] = split_result["test"]
    split_sizes = {k: len(v) for k, v in res.items()}
    logger.info(f"dataset size after adding the split: {split_sizes}")
    return res


def drop_splits(dataset: datasets.DatasetDict, split_names: List[str]) -> datasets.DatasetDict:
    result = datasets.DatasetDict(
        {name: ds for name, ds in dataset.items() if name not in split_names}
    )
    return result


def concat_splits(
    dataset: datasets.DatasetDict, splits: List[str], target: str
) -> datasets.DatasetDict:
    result = datasets.DatasetDict({name: ds for name, ds in dataset.items() if name not in splits})
    splits_to_concat = [dataset[name] for name in splits]
    # ensure that the document types are the same
    document_type = None
    dataset_type = None
    for split in splits_to_concat:
        if document_type is not None and split.document_type != document_type:
            raise ValueError(
                f"document types of splits to concatenate differ: {document_type} != {split.document_type}"
            )
        document_type = split.document_type
        if dataset_type is not None and type(split) != dataset_type:
            raise ValueError(
                f"dataset types of splits to concatenate differ: {dataset_type} != {type(split)}"
            )
        dataset_type = type(split)
    if document_type is None or dataset_type is None:
        raise ValueError("please provide at least one split to concatenate")
    concatenated = datasets.concatenate_datasets(splits_to_concat)
    result[target] = dataset_type.from_hf_dataset(concatenated, document_type=document_type)
    split_sizes = {k: len(v) for k, v in result.items()}
    logger.info(f"dataset size after concatenating splits: {split_sizes}")
    return result


def filter(
    dataset: datasets.DatasetDict,
    split: str,
    function: Optional[Union[Callable, str]] = None,
    result_split_name: Optional[str] = None,
    **kwargs,
) -> datasets.DatasetDict:
    if function is not None:
        # create a shallow copy to not modify the input
        dataset = type(dataset)(dataset)
        if isinstance(function, str):
            function = _resolve_target(function, full_key="")

        pie_split = dataset[split]
        if isinstance(pie_split, Dataset):
            hf_split = datasets.Dataset(**Dataset.get_base_kwargs(pie_split))
        elif isinstance(pie_split, IterableDataset):
            hf_split = datasets.IterableDataset(**IterableDataset.get_base_kwargs(pie_split))
        else:
            raise Exception(f"dataset split has unknown type: {type(pie_split)}")
        hf_split_filtered = hf_split.filter(function=function, **kwargs)
        target_split_name = result_split_name or split
        dataset[target_split_name] = type(pie_split).from_hf_dataset(
            dataset=hf_split_filtered, document_type=pie_split.document_type
        )
        logger.info(
            f"filtered split [{target_split_name}] has {len(dataset[target_split_name])} entries"
        )
    return dataset


def move_to_new_split(
    dataset: datasets.DatasetDict,
    ids: List[str],
    source_split: str = "train",
    target_split: str = "test",
) -> datasets.DatasetDict:

    ids_set = set(ids)
    dataset_without_ids = filter(
        dataset=dataset,
        split=source_split,
        function=lambda ex: ex["id"] not in ids_set,
    )
    dataset_with_only_ids = filter(
        dataset=dataset,
        split=source_split,
        function=lambda ex: ex["id"] in ids_set,
    )
    dataset_without_ids[target_split] = dataset_with_only_ids[source_split]

    split_sizes = {k: len(v) for k, v in dataset_without_ids.items()}
    logger.info(f"dataset size after moving to new split: {split_sizes}")
    return dataset_without_ids


def cast_document_type(
    dataset: datasets.DatasetDict, new_document_type: Union[Type[Document], str], **kwargs
) -> datasets.DatasetDict:
    if isinstance(new_document_type, str):
        new_document_type = _resolve_target(new_document_type, full_key="")

    result = datasets.DatasetDict(
        {
            name: ds.cast_document_type(new_document_type=new_document_type, **kwargs)
            for name, ds in dataset.items()
        }
    )
    return result
