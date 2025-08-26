import os

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.serializer.interface import DocumentSerializer
from src.train import train
from tests.helpers.run_if import RunIf


def test_train_fast_dev_run(cfg_train):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
    train(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_fast_dev_run_gpu(cfg_train):
    """Run for 1 train, val and test step on GPU."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
    train(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train):
    """Train 1 epoch on GPU with mixed-precision."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.precision = 16
    train(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train):
    """Train 1 epoch with validation loop twice per epoch."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.slow
@pytest.mark.veryslow
def test_train_ddp_sim(cfg_train):
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp_spawn"
        cfg_train.trainer.limit_train_batches = 3
        cfg_train.trainer.limit_val_batches = 3
    train(cfg_train)


@pytest.mark.slow
@pytest.mark.veryslow
def test_train_resume(tmp_path, cfg_train):
    """Run 1 epoch, finish, and resume for another epoch."""
    with open_dict(cfg_train):
        cfg_train.seed = 12345
        cfg_train.trainer.max_epochs = 1
        cfg_train.callbacks.model_checkpoint.save_last = True
        cfg_train.trainer.limit_train_batches = 3
        cfg_train.trainer.limit_val_batches = 3

    HydraConfig().set_config(cfg_train)
    metric_dict_1, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files

    # note: all values set above are still valid, if not overwritten here
    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    metric_dict_2, _ = train(cfg_train)

    assert metric_dict_2["train/loss"] < metric_dict_1["train/loss"]


@pytest.mark.slow
def test_train_val_predict(cfg_train, tmp_path):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.seed = 12345
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.limit_train_batches = 2
        cfg_train.trainer.limit_val_batches = 2
        # only predict the first two batches
        cfg_train.pipeline.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"

        cfg_train.validate = True
        cfg_train.predict = True

    result_dict, object_dict = train(cfg_train)
    # assert that the serializer returned the same path as the one in the config
    assert result_dict["serializer/path"] == cfg_train.paths.prediction_save_dir
    # assert that the serializer returned the same split as the test split used in the datamodule
    assert result_dict["serializer/split"] == object_dict["trainer"].datamodule.test_split
    # get the serializer from the object dict
    serializer: DocumentSerializer = object_dict["serializer"]
    annotated_documents = serializer.read(
        path=result_dict["serializer/path"], split=result_dict["serializer/split"]
    )
    # get the first document
    assert len(annotated_documents) > 0
    doc = annotated_documents[0]
    # Check that the document has predictions
    assert len(doc.labeled_spans.predictions) > 0


@pytest.mark.slow
def test_eval(tmp_path, cfg_eval):

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = train(cfg_eval)

    expected = {"test/f1": torch.tensor(0.96206), "test/loss": torch.tensor(0.005253)}
    torch.testing.assert_close(test_metric_dict, expected)


@pytest.mark.slow
def test_train_eval(tmp_path, cfg_train, cfg_eval):
    """Train for 1 epoch with `train.py` and evaluate with `eval.py`"""
    assert str(tmp_path) == cfg_train.paths.save_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True
        cfg_train.trainer.limit_train_batches = 10
        # ensure reproducibility
        cfg_train.seed = 42
        cfg_train.trainer.deterministic = True

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)
    assert train_metric_dict["train/f1"] > 0.0

    assert os.path.exists(cfg_train.paths.model_save_dir)

    with open_dict(cfg_eval):
        cfg_eval.model_name_or_path = cfg_train.paths.model_save_dir

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = train(cfg_eval)

    assert len(test_metric_dict) > 0
    for k in test_metric_dict:
        assert torch.isclose(train_metric_dict[k], test_metric_dict[k])
