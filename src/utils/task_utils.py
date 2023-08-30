import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Dict

from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only

from src.utils.logging_utils import close_loggers, get_pylogger
from src.utils.rich_utils import enforce_tags, print_config_tree

log = get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # apply extra utilities
        extras(cfg)

        # execute the task
        start_time = time.time()
        try:
            task_result = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.pipeline_type}' execution time: {time.time() - start_time} (s)"
            save_file(path, content)  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return task_result

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def load_value_from_file(path: str, split_path_key: str = ":", split_key_parts: str = "/") -> Dict:
    """Load a value from a file. The path can point to elements within the file (see split_path_key
    parameter) and that can be nested (see split_key_parts parameter). For now, only .json files
    are supported.

    Args:
        path: path to the file (and data within the file)
        split_path_key: split the path on this value to get the path to the file and the key within the file
        split_key_parts: the value to split the key on to get the nested keys
    """

    parts_path = path.split(split_path_key, maxsplit=1)
    file_extension = os.path.splitext(parts_path[0])[1]
    if file_extension == ".json":
        with open(parts_path[0], "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Expected .json file, got {file_extension}")

    if len(parts_path) == 1:
        return data

    keys = parts_path[1].split(split_key_parts)
    for key in keys:
        data = data[key]
    return data


def replace_sys_args_with_values_from_files(
    load_prefix: str = "LOAD_ARG:",
    load_multi_prefix: str = "LOAD_MULTI_ARG:",
    **load_value_from_file_kwargs,
) -> None:
    """Replaces arguments in sys.argv with values loaded from files.

    Examples:
        # config.json contains {"a": 1, "b": 2}
        python train.py LOAD_ARG:job_return_value.json
        # this will pass "{a:1,b:2}" as the first argument to train.py

        # config.json contains [1, 2, 3]
        python train.py LOAD_MULTI_ARG:job_return_value.json
        # this will pass "1,2,3" as the first argument to train.py

        # config.json contains {"model": {"ouput_dir": ["path1", "path2"], f1: [0.7, 0.6]}}
        python train.py load_model=LOAD_ARG:job_return_value.json:model/output_dir
        # this will pass "load_model=path1,path2" to train.py

    Args:
        load_prefix: the prefix to use for loading a single value from a file
        load_multi_prefix: the prefix to use for loading a list of values from a file
        **load_value_from_file_kwargs: additional kwargs to pass to load_value_from_file
    """

    updated_args = []
    for arg in sys.argv[1:]:
        is_multirun_arg = False
        if load_prefix in arg:
            parts = arg.split(load_prefix, maxsplit=1)
        elif load_multi_prefix in arg:
            parts = arg.split(load_multi_prefix, maxsplit=1)
            is_multirun_arg = True
        else:
            updated_args.append(arg)
            continue
        if len(parts) == 2:
            log.warning(f'Replacing argument value for "{parts[0]}" with content from {parts[1]}')
            json_value = load_value_from_file(parts[1], **load_value_from_file_kwargs)
            json_value_str = json.dumps(json_value)
            # replace quotes and spaces
            json_value_str = json_value_str.replace('"', "").replace(" ", "")
            # remove outer brackets
            if is_multirun_arg:
                if not isinstance(json_value, list):
                    raise ValueError(
                        f"Expected list for multirun argument, got {type(json_value)}. If you just want "
                        f"to set a single value, use {load_prefix} instead of {load_multi_prefix}."
                    )
                json_value_str = json_value_str[1:-1]
            # add outer quotes
            modified_arg = f"{parts[0]}{json_value_str}"
            updated_args.append(modified_arg)
        else:
            updated_args.append(arg)
    # Set sys.argv to the updated arguments
    sys.argv = [sys.argv[0]] + updated_args
