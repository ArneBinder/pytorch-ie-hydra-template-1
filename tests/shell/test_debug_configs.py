import pytest

from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_debug_default():
    command = ["train.py", "debug=default", "++trainer.gpus=1"]
    run_command(command)


def test_debug_limit_batches():
    command = ["train.py", "debug=limit_batches"]
    run_command(command)


def test_debug_overfit():
    command = ["train.py", "debug=overfit"]
    run_command(command)


@pytest.mark.slow
def test_debug_profiler():
    command = ["train.py", "debug=profiler"]
    run_command(command)


def test_debug_step():
    command = ["train.py", "debug=step"]
    run_command(command)


def test_debug_test_only():
    command = ["train.py", "debug=test_only"]
    run_command(command)
