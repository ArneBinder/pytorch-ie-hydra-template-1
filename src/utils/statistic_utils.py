from collections import defaultdict
from math import sqrt
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from pytorch_ie import DatasetDict
from pytorch_ie.core import Document
from pytorch_ie.utils.hydra import resolve_target
from typing_extensions import TypeAlias

from src.metrics.interface import DocumentMetric
from src.metrics.statistics import flatten_dict
from src.utils.logging_utils import get_pylogger

logger = get_pylogger(__name__)


def _variance(data, ddof=0):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - ddof)


def _stddev(data):
    var = _variance(data)
    std_dev = sqrt(var)
    return std_dev


BaseType: TypeAlias = Union[int, float]


def _aggregate_histogram_values(values: List[BaseType], entries: Iterable[str]):
    res: Dict[str, Union[int, float]] = {}
    if len(values) == 0:
        return res
    for entry in entries:
        if entry == "mean":
            res["mean"] = sum(values) / len(values)
        elif entry == "min":
            res["min"] = min(values)
        elif entry == "max":
            res["max"] = max(values)
        elif entry == "num":
            res["num"] = len(values)
        elif entry == "var":
            res["var"] = _variance(values)
        elif entry == "stddev":
            res["stddev"] = _stddev(values)
        elif entry == "sum":
            res["sum"] = sum(values)
        else:
            raise ValueError(f"unknown aggregation entry: {entry}")
    return res


def _stats_as_df(
    stats: Dict[str, Union[Dict[str, BaseType], List[BaseType]]],
    aggregate_functions: Iterable[str],
    key_names: Optional[Tuple[str, ...]] = None,
):
    # aggregate lists, if necessary
    stats = {
        k: _aggregate_histogram_values(values, entries=aggregate_functions)
        if isinstance(values, list)
        else values
        for k, values in stats.items()
    }

    df = pd.DataFrame.from_dict(stats).sort_index(axis="index").sort_index(axis="columns")
    if key_names is not None:
        if len(key_names) < 1:
            raise ValueError("less than one key name is not allowed")
        if len(key_names) > 1:
            df.columns.names = key_names
        else:
            df.columns.name = key_names[0]
    return df


DEFAULT_AGGREGATE_FUNCTIONS = ("mean", "stddev", "min", "max")


def show_statistics(
    data,
    title: str,
    format: str = "markdown",
    is_histogram_data: bool = False,
    key_names: Optional[Tuple[str, ...]] = None,
    group_by_key: Optional[Union[str, int, List[Union[str, int]]]] = None,
    aggregate_functions: Optional[Iterable[str]] = None,
    **kwargs_show,
):

    if group_by_key is not None:
        if isinstance(group_by_key, list):
            group_by_key_list = group_by_key
        else:
            group_by_key_list = [group_by_key]
        if len(set(group_by_key_list)) < len(group_by_key_list):
            raise ValueError(f"group_by_key contains duplicated entries: {group_by_key_list}")
        if len(group_by_key_list) < 1:
            raise ValueError(
                f"group_by_key needs to consist of at least one entry, but it is: {group_by_key_list}"
            )
        group_by_indices: List[int]
        if all(isinstance(gbk, int) for gbk in group_by_key_list):
            group_by_indices = group_by_key_list  # type: ignore
        else:
            if key_names is None:
                raise ValueError(
                    f"key_names is required if group_by_key [{group_by_key_list}] contains any strings, "
                    f"but key_names is None"
                )
            group_by_indices = [
                key_names.index(gbk) if isinstance(gbk, str) else gbk for gbk in group_by_key_list
            ]
        new_key_names = (
            tuple(key_names[idx] for idx in group_by_indices) if key_names is not None else None
        )

        new_stats: Dict[Tuple[str, ...], Dict[Tuple[str, ...], BaseType]] = defaultdict(dict)
        for keys, v in data.items():
            key_group = tuple(keys[idx] for idx in group_by_indices)
            keys_other = tuple(
                keys[idx] for idx in range(len(keys)) if idx not in group_by_indices
            )
            if len(keys_other) == 1:
                keys_other = keys_other[0]
            new_stats[key_group][keys_other] = v
        for keys, v in new_stats.items():
            show_statistics(
                data=v,
                title=f"{title}: {', '.join(keys)}",
                format=format,
                is_histogram_data=is_histogram_data,
                key_names=new_key_names,
                aggregate_functions=aggregate_functions,
                **kwargs_show,
            )
    else:
        data = {k[0] if isinstance(k, tuple) and len(k) == 1 else k: v for k, v in data.items()}

        if format == "markdown":
            df = _stats_as_df(
                stats=data,
                key_names=key_names,
                aggregate_functions=aggregate_functions or DEFAULT_AGGREGATE_FUNCTIONS,
            )
            logger.info(f"{title}:\n{df.to_markdown(**kwargs_show)}")
        elif format == "json":
            df = _stats_as_df(
                stats=data,
                key_names=key_names,
                aggregate_functions=aggregate_functions or DEFAULT_AGGREGATE_FUNCTIONS,
            )
            logger.info(f"{title}:\n{df.to_json(**kwargs_show)}")
        elif format == "plot":
            import plotext as plt

            if is_histogram_data and aggregate_functions is None:
                for k, values in data.items():
                    plt.hist(values, label=k, **kwargs_show)
                plt.title(title)
                plt.show()
                plt.clear_figure()
            else:
                df = _stats_as_df(
                    stats=data,
                    key_names=key_names,
                    aggregate_functions=aggregate_functions or DEFAULT_AGGREGATE_FUNCTIONS,
                )
                plt.multiple_bar(
                    df.index, list(dict(df).values()), label=df.columns, **kwargs_show
                )
                plt.title(title)
                plt.show()
                plt.clear_figure()

        else:
            raise ValueError(f"unknown format to show statistics: {format}")


# The metric should return a single int or float or a list of such values ...
ResultTerminal: TypeAlias = Union[BaseType, List[BaseType]]
# ... or such entries nested arbitrarily deep inside dictionaries.
ResultDict: TypeAlias = Dict[str, Union[ResultTerminal, "ResultDict"]]


def prepare_data(
    dataset: DatasetDict,
    metric: Union[str, Callable[[Document], Union[ResultTerminal, ResultDict]], DocumentMetric],
) -> Dict[Tuple[str, ...], List[BaseType]]:
    if isinstance(metric, DocumentMetric):
        nested_values = metric(dataset)
        return flatten_dict(nested_values)
    else:
        metric_func = resolve_target(metric)
        stats = defaultdict(list)
        for s_name, split in dataset.items():
            for doc in split:
                metric_result = metric_func(doc)
                if isinstance(metric_result, dict):
                    metric_result_flat = flatten_dict(metric_result)
                    for k, v in metric_result_flat.items():
                        if isinstance(v, list):
                            stats[(s_name,) + k].extend(v)
                        else:
                            stats[(s_name,) + k].append(v)
                else:
                    if isinstance(metric_result, list):
                        stats[(s_name,)].extend(metric_result)
                    else:
                        stats[(s_name,)].append(metric_result)
        return dict(stats)


def collect_statistics(
    dataset: DatasetDict,
    metric: Union[str, Callable[[Document], Union[ResultTerminal, ResultDict]], DocumentMetric],
    title: str,
    group_by_key: Optional[Union[str, int, List[Union[str, int]]]] = None,
    key_names: Optional[Tuple[str, ...]] = None,
    aggregate_functions: Optional[Iterable[str]] = None,
    **kwargs_show,
) -> None:
    data = prepare_data(dataset=dataset, metric=metric)

    is_histogram_data = False
    num_keys = None
    for k, v in data.items():
        if key_names is not None and len(k) != len(key_names):
            raise ValueError(
                f"number of key levels (key: {k}) does not match key names: {key_names}"
            )
        if num_keys is None:
            num_keys = len(k)
        elif num_keys != len(k):
            raise ValueError("keys need to have the same number of levels ")
        if isinstance(v, list):
            is_histogram_data = True
    if group_by_key is None and num_keys is not None and num_keys > 1:
        group_by_key = list(range(num_keys - 1))
    stats_sorted = dict(sorted(data.items()))
    show_statistics(
        data=stats_sorted,
        title=title,
        is_histogram_data=is_histogram_data,
        aggregate_functions=aggregate_functions,
        key_names=key_names,
        group_by_key=group_by_key,
        **kwargs_show,
    )
