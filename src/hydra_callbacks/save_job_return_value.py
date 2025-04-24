import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig


def to_py_obj(obj):
    """Convert a PyTorch tensor, Numpy array or python list to a python list.

    Modified version of transformers.utils.generic.to_py_obj.
    """
    if isinstance(obj, dict):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_py_obj(o) for o in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, (np.ndarray, np.number)):  # tolist also works on 0d np arrays
        return obj.tolist()
    else:
        return obj


def list_of_dicts_to_dict_of_lists_recursive(list_of_dicts):
    """Convert a list of dicts to a dict of lists recursively.

    Examples:
        # works with nested dicts
        >>> list_of_dicts_to_dict_of_lists_recursive([{"a": 1, "b": {"c": 2}}, {"a": 3, "b": {"c": 4}}])
        {'a': [1, 3], 'b': {'c': [2, 4]}}
        # works with incomplete dicts
        >>> list_of_dicts_to_dict_of_lists_recursive([{"a": 1, "b": 2}, {"a": 3}])
        {'a': [1, 3], 'b': [2, None]}

        # works with nested incomplete dicts
        >>> list_of_dicts_to_dict_of_lists_recursive([{"a": 1, "b": {"c": 2}}, {"a": 3}])
        {'a': [1, 3], 'b': {'c': [2, None]}}

        # works with nested incomplete dicts with None values
        >>> list_of_dicts_to_dict_of_lists_recursive([{"a": 1, "b": {"c": 2}}, {"a": None}])
        {'a': [1, None], 'b': {'c': [2, None]}}

    Args:
        list_of_dicts (List[dict]): A list of dicts.

    Returns:
        dict: An arbitrarily nested dict of lists.
    """
    if not list_of_dicts:
        return {}

    # Check if all elements are either None or dictionaries
    if all(d is None or isinstance(d, dict) for d in list_of_dicts):
        # Gather all keys from non-None dictionaries
        keys = set()
        for d in list_of_dicts:
            if d is not None:
                keys.update(d.keys())

        # Build up the result recursively
        return {
            k: list_of_dicts_to_dict_of_lists_recursive(
                [(d[k] if d is not None and k in d else None) for d in list_of_dicts]
            )
            for k in keys
        }
    else:
        # If items are not all dict/None, just return the list as is (base case).
        return list_of_dicts


def _flatten_dict_gen(d, parent_key: Tuple[str, ...] = ()) -> Generator:
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            yield from dict(_flatten_dict_gen(v, new_key)).items()
        else:
            yield new_key, v


def flatten_dict(d: Dict[str, Any], pad_keys: bool = True) -> Dict[Tuple[str, ...], Any]:
    """Flattens a dictionary with nested keys. Per default, the keys are padded with np.nan to have
    the same length.

    Example:
        >>> d = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
        >>> flatten_dict(d)
        {('a', 'b', 'c'): 1, ('a', 'b', 'd'): 2, ('a', 'e', np.nan): 3}

        # with padding the keys
        >>> d = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
        >>> flatten_dict(d, pad_keys=False)
        {('a', 'b', 'c'): 1, ('a', 'b', 'd'): 2, ('a', 'e'): 3}
    """
    result = dict(_flatten_dict_gen(d))
    # pad the keys with np.nan to have the same length. We use np.nan to be pandas-friendly.
    if pad_keys:
        max_num_keys = max(len(k) for k in result.keys())
        result = {
            tuple(list(k) + [np.nan] * (max_num_keys - len(k))): v for k, v in result.items()
        }
    return result


def unflatten_dict(
    d: Dict[Tuple[str, ...], Any], unpad_keys: bool = True
) -> Union[Dict[str, Any], Any]:
    """Unflattens a dictionary with nested keys. Per default, the keys are unpadded by removing
    np.nan values.

    Example:
        >>> d = {("a", "b", "c"): 1, ("a", "b", "d"): 2, ("a", "e"): 3}
        >>> unflatten_dict(d)
        {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}

        # with unpad the keys
        >>> d = {("a", "b", "c"): 1, ("a", "b", "d"): 2, ("a", "e", np.nan): 3}
        >>> unflatten_dict(d)
        {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
    """
    result: Dict[str, Any] = {}
    for k, v in d.items():
        if unpad_keys:
            k = tuple([ki for ki in k if not pd.isna(ki)])
        if len(k) == 0:
            if len(result) > 1:
                raise ValueError("Cannot unflatten dictionary with multiple root keys.")
            return v
        current = result
        for key in k[:-1]:
            current = current.setdefault(key, {})
        current[k[-1]] = v
    return result


def overrides_to_identifiers(overrides_per_result: List[List[str]], sep: str = "-") -> List[str]:
    """Converts a list of lists of overrides to a list of identifiers. But takes only the overrides
    into account, that are not identical for all results.

    Example:
        >>> overrides_per_result = [
        ...     ["a=1", "b=2", "c=3"],
        ...     ["a=1", "b=2", "c=4"],
        ...     ["a=1", "b=3", "c=3"],
        ]
        >>> overrides_to_identifiers(overrides_per_result)
        ['b=2-c=3', 'b=2-c=4', 'b=3-c=3']

    Args:
        overrides_per_result (List[List[str]]): A list of lists of overrides.
        sep (str, optional): The separator to use between the overrides. Defaults to "-".

    Returns:
        List[str]: A list of identifiers.
    """
    # get the overrides that are not identical for all results
    overrides_per_result_transposed = np.array(overrides_per_result).T.tolist()
    indices = [
        i for i, entries in enumerate(overrides_per_result_transposed) if len(set(entries)) > 1
    ]
    # convert the overrides to identifiers
    identifiers = [
        sep.join([overrides[idx] for idx in indices]) for overrides in overrides_per_result
    ]
    return identifiers


class SaveJobReturnValueCallback(Callback):
    """Save the job return-value in ${output_dir}/{job_return_value_filename}.

    This also works for multi-runs (e.g. sweeps for hyperparameter search). In this case, the result will be saved
    additionally in a common file in the multi-run log directory. If integrate_multirun_result=True, the
    job return-values are also aggregated (e.g. mean, min, max) and saved in another file.

    params:
    -------
    filenames: str or List[str] (default: "job_return_value.json")
        The filename(s) of the file(s) to save the job return-value to. If it ends with ".json",
        the return-value will be saved as a json file. If it ends with ".pkl", the return-value will be
        saved as a pickle file, if it ends with ".md", the return-value will be saved as a markdown file.
    integrate_multirun_result: bool (default: True)
        If True, the job return-values of all jobs from a multi-run will be rearranged into a dict of lists (maybe
        nested), where the keys are the keys of the job return-values and the values are lists of the corresponding
        values of all jobs. This is useful if you want to access specific values of all jobs in a multi-run all at once.
        Also, aggregated values (e.g. mean, min, max) are created for all numeric values and saved in another file.
    multirun_aggregator_blacklist: List[str] (default: None)
        A list of keys to exclude from the aggregation (of multirun results), such as "count" or "25%". If None,
        all keys are included. See pd.DataFrame.describe() for possible aggregation keys.
        For numeric values, it is recommended to use ["min", "25%", "50%", "75%", "max"]
        which will result in keeping only the count, mean and std values.
    sort_markdown_columns: bool (default: False)
        If True, the columns of the markdown table are sorted alphabetically.
    multirun_create_ids_from_overrides: bool (default: True)
        Create job identifiers from the overrides of the jobs in a multi-run. If False, the job index is used as
        identifier.
    markdown_round_digits: int (default: 3)
        The number of digits to round the values in the markdown file. If None, no rounding is applied.
    multirun_job_id_key: str (default: "job_id")
        The key to use for the job identifiers in the integrated multi-run result.
    paths_file: str (default: None)
        The file to save the paths of the log directories to. If None, the paths are not saved.
    path_id: str (default: None)
        A prefix to add to each line in the paths_file separated by a colon. If None, no prefix is added.
    multirun_paths_file: str (default: None)
        The file to save the paths of the multi-run log directories to. If None, the paths are not saved.
    multirun_path_id: str (default: None)
        A prefix to add to each line in the multirun_paths_file separated by a colon. If None, no prefix is added.
    """

    def __init__(
        self,
        filenames: Union[str, List[str]] = "job_return_value.json",
        integrate_multirun_result: bool = False,
        multirun_aggregator_blacklist: Optional[List[str]] = None,
        sort_markdown_columns: bool = False,
        multirun_create_ids_from_overrides: bool = True,
        markdown_round_digits: Optional[int] = 3,
        multirun_job_id_key: str = "job_id",
        paths_file: Optional[str] = None,
        path_id: Optional[str] = None,
        multirun_paths_file: Optional[str] = None,
        multirun_path_id: Optional[str] = None,
    ) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.filenames = [filenames] if isinstance(filenames, str) else filenames
        self.integrate_multirun_result = integrate_multirun_result
        self.job_returns: List[JobReturn] = []
        self.multirun_aggregator_blacklist = multirun_aggregator_blacklist
        self.sort_markdown_columns = sort_markdown_columns
        self.multirun_create_ids_from_overrides = multirun_create_ids_from_overrides
        self.multirun_job_id_key = multirun_job_id_key
        self.markdown_round_digits = markdown_round_digits
        self.multirun_paths_file = multirun_paths_file
        self.multirun_path_id = multirun_path_id
        self.paths_file = paths_file
        self.path_id = path_id

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        self.job_returns.append(job_return)
        output_dir = Path(config.hydra.runtime.output_dir)  # / Path(config.hydra.output_subdir)
        if self.paths_file is not None:
            # append the output_dir to the file
            with open(self.paths_file, "a") as file:
                file.write(f"{output_dir}\n")

        for filename in self.filenames:
            self._save(obj=job_return.return_value, filename=filename, output_dir=output_dir)

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        job_ids: Union[List[str], List[int]]
        if self.multirun_create_ids_from_overrides:
            job_ids = overrides_to_identifiers([jr.overrides for jr in self.job_returns])
        else:
            job_ids = list(range(len(self.job_returns)))

        if self.integrate_multirun_result:
            # rearrange the job return-values of all jobs from a multi-run into a dict of lists (maybe nested),
            obj = list_of_dicts_to_dict_of_lists_recursive(
                [jr.return_value for jr in self.job_returns]
            )
            if not isinstance(obj, dict):
                obj = {"value": obj}
            if self.multirun_create_ids_from_overrides:
                obj[self.multirun_job_id_key] = job_ids

            # also create an aggregated result
            # convert to python object to allow selecting numeric columns
            obj_py = to_py_obj(obj)
            obj_flat = flatten_dict(obj_py)
            # create dataframe from flattened dict
            df_flat = pd.DataFrame(obj_flat)
            # select only the numeric values
            df_numbers_only = df_flat.select_dtypes(["number"])
            cols_removed = set(df_flat.columns) - set(df_numbers_only.columns)
            if len(cols_removed) > 0:
                self.log.warning(
                    f"Removed the following columns from the aggregated result because they are not numeric: "
                    f"{cols_removed}"
                )
            if len(df_numbers_only.columns) == 0:
                obj_aggregated = None
            else:
                # aggregate the numeric values
                df_described = df_numbers_only.describe()
                # remove rows in the blacklist
                if self.multirun_aggregator_blacklist is not None:
                    df_described = df_described.drop(
                        self.multirun_aggregator_blacklist, errors="ignore", axis="index"
                    )
                # add the aggregation keys (e.g. mean, min, ...) as most inner keys and convert back to dict
                obj_flat_aggregated = df_described.T.stack().to_dict()
                # unflatten because _save() works better with nested dicts
                obj_aggregated = unflatten_dict(obj_flat_aggregated)
        else:
            # create a dict of the job return-values of all jobs from a multi-run
            # (_save() works better with nested dicts)
            obj = {
                identifier: jr.return_value for identifier, jr in zip(job_ids, self.job_returns)
            }
            obj_aggregated = None
        output_dir = Path(config.hydra.sweep.dir)
        if self.multirun_paths_file is not None:
            # append the output_dir to the file
            line = f"{output_dir}\n"
            if self.multirun_path_id is not None:
                line = f"{self.multirun_path_id}:{line}"
            with open(self.multirun_paths_file, "a") as file:
                file.write(line)

        for filename in self.filenames:
            self._save(
                obj=obj,
                filename=filename,
                output_dir=output_dir,
                is_tabular_data=self.integrate_multirun_result,
            )
            # if available, also save the aggregated result
            if obj_aggregated is not None:
                file_base_name, ext = os.path.splitext(filename)
                filename_aggregated = f"{file_base_name}.aggregated{ext}"
                self._save(
                    obj=obj_aggregated,
                    filename=filename_aggregated,
                    output_dir=output_dir,
                    # If we have aggregated (integrated multi-run) results, we unstack the last level,
                    # i.e. the aggregation key.
                    unstack_last_index_level=True,
                )

    def _save(
        self,
        obj: Any,
        filename: str,
        output_dir: Path,
        is_tabular_data: bool = False,
        unstack_last_index_level: bool = False,
    ) -> None:
        self.log.info(f"Saving job_return in {output_dir / filename}")
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir is not None
        if filename.endswith(".pkl"):
            with open(str(output_dir / filename), "wb") as file:
                pickle.dump(obj, file, protocol=4)
        elif filename.endswith(".json"):
            # Convert PyTorch tensors and numpy arrays to native python types
            obj_py = to_py_obj(obj)
            with open(str(output_dir / filename), "w") as file:
                json.dump(obj_py, file, indent=2)
        elif filename.endswith(".md"):
            # Convert PyTorch tensors and numpy arrays to native python types
            obj_py = to_py_obj(obj)
            if not isinstance(obj_py, dict):
                obj_py = {"value": obj_py}
            obj_py_flat = flatten_dict(obj_py)

            if is_tabular_data:
                # In the case of (not aggregated) integrated multi-run result, we expect to have
                # multiple values for each key. We therefore just convert the dict to a pandas DataFrame.
                result = pd.DataFrame(obj_py_flat)
                job_id_column = (self.multirun_job_id_key,) + (np.nan,) * (
                    result.columns.nlevels - 1
                )
                if job_id_column in result.columns:
                    result = result.set_index(job_id_column)
                    result.index.name = self.multirun_job_id_key
            else:
                # Otherwise, we have only one value for each key. We convert the dict to a pandas Series.
                series = pd.Series(obj_py_flat)
                # The series has a MultiIndex because flatten_dict() uses a tuple as key.
                if len(series.index.levels) <= 1:
                    # If there is only one level, we just use the first level values as index.
                    series.index = series.index.get_level_values(0)
                    result = series
                else:
                    # If there are multiple levels, we unstack the series to get a DataFrame
                    # providing a better overview.
                    if unstack_last_index_level:
                        # If we have aggregated (integrated multi-run) results, we unstack the last level,
                        # i.e. the aggregation key.
                        result = series.unstack(-1)
                    else:
                        # Otherwise we have a default multi-run result and unstack the first level,
                        # i.e. the identifier created from the overrides, and transpose the result
                        # to have the individual jobs as rows.
                        result = series.unstack(0).T

            if isinstance(result, pd.DataFrame) and self.sort_markdown_columns:
                result = result.sort_index(axis=1)

            if self.markdown_round_digits is not None:
                result = result.round(self.markdown_round_digits)

            with open(str(output_dir / filename), "w") as file:
                file.write(result.to_markdown())

        else:
            raise ValueError("Unknown file extension")
