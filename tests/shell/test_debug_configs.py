import pytest

from tests.helpers.run_command import run_command


@pytest.mark.slow
def test_debug_default():
    command = ["train.py", "debug=default",  "++trainer.limit_train_batches=5"]
    run_command(command)


def test_debug_limit_batches():
    command = ["train.py", "debug=limit_batches"]
    run_command(command)


def test_debug_overfit():
    command = ["train.py", "debug=overfit", "trainer.max_epochs=5"]
    run_command(command)


@pytest.mark.slow
def test_debug_profiler():
    command = ["train.py", "debug=profiler", "++trainer.limit_train_batches=5"]
    run_command(command)


def test_debug_step():
    command = ["train.py", "debug=step"]
    run_command(command)


def test_debug_test_only():
    command = ["train.py", "debug=test_only"]
    run_command(command)
