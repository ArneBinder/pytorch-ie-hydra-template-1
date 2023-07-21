import json
import logging
import pickle
from pathlib import Path
from typing import Any, List

import numpy as np
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


class SaveJobReturnValueCallback(Callback):
    """Save the job return-value in ${output_dir}/{job_return_value_filename}.

    This also works for multi-runs (sweeps).

    params:
    -------
    job_return_value_filename: str
        The filename of the job return-value (default: "job_return_value.json"). If it ends with ".json",
        the return-value will be saved as a json file. If it ends with ".pkl", the return-value will be
        saved as a pickle file.
    merge_multirun_result: bool (default: True)
        If True, the job return-values of all jobs from a multi-run will be rearranged into a dict of lists (maybe
        nested), where the keys are the keys of the job return-values and the values are lists of the corresponding
        values of all jobs. This is useful if you want to access specific values of all jobs in a multi-run all at once.
    """

    def __init__(
        self,
        job_return_value_filename: str = "job_return_value.json",
        merge_multirun_result: bool = True,
    ) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.job_return_value_filename = job_return_value_filename
        self.merge_multirun_result = merge_multirun_result
        self.job_returns: List[JobReturn] = []

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        self.job_returns.append(job_return)
        output_dir = Path(config.hydra.runtime.output_dir)  # / Path(config.hydra.output_subdir)
        filename = self.job_return_value_filename
        self._save(obj=job_return.return_value, filename=filename, output_dir=output_dir)
        self.log.info(f"Saving job_return in {output_dir / filename}")

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        filename = self.job_return_value_filename
        output_dir = Path(config.hydra.sweep.dir)
        obj = [jr.return_value for jr in self.job_returns]
        if self.merge_multirun_result:
            obj = list_of_dicts_to_dict_of_lists_recursive(obj)
        self._save(obj=obj, filename=filename, output_dir=output_dir)
        self.log.info(f"Saving job_return in {output_dir / filename}")

    def _save(self, obj: Any, filename: str, output_dir: Path) -> None:
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
        else:
            raise ValueError("Unknown file extension")
