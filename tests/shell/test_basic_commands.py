import os.path

import pytest

from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf

"""
A couple of sanity checks to make sure the model doesn't crash with different running options.
"""


def test_fast_dev_run_with_evaluation():
    """Test running for 1 train, val and test batch."""
    command = ["train.py", "++trainer.fast_dev_run=true", "++test=true"]
    run_command(command)


def test_evaluation_single_batch():
    """Test the test script with a single batch."""
    command = ["evaluate.py", "++trainer.limit_test_batches=1"]
    run_command(command)


def test_prediction_fast_dev_run(tmp_path):
    """Test the prediction script with two input encodings (pipeline.fast_dev_run)."""
    out_path = tmp_path / "predictions"
    command = ["predict.py", f"out_path={out_path}", "++pipeline.fast_dev_run=true"]
    run_command(command)
    assert os.path.exists(f"{out_path}.jsonl")


@pytest.mark.skip(reason="this takes too much time")
@pytest.mark.slow
def test_cpu():
    """Test running 1 epoch on CPU."""
    command = ["train.py", "++trainer.max_epochs=1", "++trainer.gpus=0"]
    run_command(command)


# use RunIf to skip execution of some tests, e.g. when no gpus are available
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_gpu():
    """Test running 1 epoch on GPU."""
    command = [
        "train.py",
        "++trainer.max_epochs=1",
        "++trainer.gpus=1",
    ]
    run_command(command)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_mixed_precision():
    """Test running 1 epoch with pytorch native automatic mixed precision (AMP)."""
    command = [
        "train.py",
        "++trainer.max_epochs=1",
        "++trainer.gpus=1",
        "++trainer.precision=16",
    ]
    run_command(command)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_double_validation_loop():
    """Test running 1 epoch with validation loop twice per epoch."""
    command = [
        "train.py",
        "++trainer.max_epochs=1",
        "++trainer.gpus=1",
        "++trainer.val_check_interval=0.5",
    ]
    run_command(command)


@pytest.mark.slow
def test_evaluation_cpu():
    """Test the test script."""
    command = ["evaluate.py", "++trainer.gpus=0", "++trainer.limit_test_batches=5"]
    run_command(command)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_evaluation_gpu():
    """Test the test script."""
    command = ["evaluate.py", "++trainer.gpus=1"]
    run_command(command)
