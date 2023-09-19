import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple, Union

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

    Example:
        # works with nested dicts
        >>> list_of_dicts_to_dict_of_lists_recursive([{"a": 1, "b": {"c": 2}}, {"a": 3, "b": {"c": 4}}])
        {'b': {'c': [2, 4]}, 'a': [1, 3]}
        # works with incomplete dicts
        >>> list_of_dicts_to_dict_of_lists_recursive([{"a": 1, "b": 2}, {"a": 3}])
        {'b': [2, None], 'a': [1, 3]}

    Args:
        list_of_dicts (List[dict]): A list of dicts.

    Returns:
        dict: A dict of lists.
    """
    if isinstance(list_of_dicts, list):
        if len(list_of_dicts) == 0:
            return {}
        elif isinstance(list_of_dicts[0], dict):
            keys = set()
            for d in list_of_dicts:
                if not isinstance(d, dict):
                    raise ValueError("Not all elements of the list are dicts.")
                keys.update(d.keys())
            return {
                k: list_of_dicts_to_dict_of_lists_recursive(
                    [d.get(k, None) for d in list_of_dicts]
                )
                for k in keys
            }
        else:
            return list_of_dicts
    else:
        return list_of_dicts


def _flatten_dict_gen(d, parent_key: Tuple[str, ...] = ()) -> Generator:
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            yield from dict(_flatten_dict_gen(v, new_key)).items()
        else:
            yield new_key, v


def flatten_dict(d: Dict[str, Any]) -> Dict[Tuple[str, ...], Any]:
    return dict(_flatten_dict_gen(d))


def unflatten_dict(d: Dict[Tuple[str, ...], Any]) -> Union[Dict[str, Any], Any]:
    """Unflattens a dictionary with nested keys.

    Example:
        >>> d = {("a", "b", "c"): 1, ("a", "b", "d"): 2, ("a", "e"): 3}
        >>> unflatten_dict(d)
        {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
    """
    result: Dict[str, Any] = {}
    for k, v in d.items():
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
    """

    def __init__(
        self,
        filenames: Union[str, List[str]] = "job_return_value.json",
        integrate_multirun_result: bool = False,
    ) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.filenames = [filenames] if isinstance(filenames, str) else filenames
        self.integrate_multirun_result = integrate_multirun_result
        self.job_returns: List[JobReturn] = []

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        self.job_returns.append(job_return)
        output_dir = Path(config.hydra.runtime.output_dir)  # / Path(config.hydra.output_subdir)
        for filename in self.filenames:
            self._save(obj=job_return.return_value, filename=filename, output_dir=output_dir)

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        if self.integrate_multirun_result:
            # rearrange the job return-values of all jobs from a multi-run into a dict of lists (maybe nested),
            obj = list_of_dicts_to_dict_of_lists_recursive(
                [jr.return_value for jr in self.job_returns]
            )
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
                # add the aggregation keys (e.g. mean, min, ...) as most inner keys and convert back to dict
                obj_flat_aggregated = df_described.T.stack().to_dict()
                # unflatten because _save() works better with nested dicts
                obj_aggregated = unflatten_dict(obj_flat_aggregated)
        else:
            # create a dict of the job return-values of all jobs from a multi-run
            # (_save() works better with nested dicts)
            ids = overrides_to_identifiers([jr.overrides for jr in self.job_returns])
            obj = {identifier: jr.return_value for identifier, jr in zip(ids, self.job_returns)}
            obj_aggregated = None
        output_dir = Path(config.hydra.sweep.dir)
        for filename in self.filenames:
            self._save(
                obj=obj,
                filename=filename,
                output_dir=output_dir,
                multi_run_result=self.integrate_multirun_result,
            )
            # if available, also save the aggregated result
            if obj_aggregated is not None:
                file_base_name, ext = os.path.splitext(filename)
                filename_aggregated = f"{file_base_name}.aggregated{ext}"
                self._save(obj=obj_aggregated, filename=filename_aggregated, output_dir=output_dir)

    def _save(
        self, obj: Any, filename: str, output_dir: Path, multi_run_result: bool = False
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
            obj_py_flat = flatten_dict(obj_py)

            if multi_run_result:
                # In the case of multi-run, we expect to have multiple values for each key.
                # We therefore just convert the dict to a pandas DataFrame.
                result = pd.DataFrame(obj_py_flat)
            else:
                # In the case of a single job, we expect to have only one value for each key.
                # We therefore convert the dict to a pandas Series and ...
                series = pd.Series(obj_py_flat)
                if len(series.index.levels) > 1:
                    # ... if the Series has multiple index levels, we create a DataFrame by unstacking the last level.
                    result = series.unstack(-1)
                else:
                    # ... otherwise we just unpack the one-entry index values and save the resulting Series.
                    series.index = series.index.get_level_values(0)
                    result = series

            with open(str(output_dir / filename), "w") as file:
                file.write(result.to_markdown())

        else:
            raise ValueError("Unknown file extension")
