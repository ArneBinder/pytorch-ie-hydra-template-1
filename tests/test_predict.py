from os import path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.predict import predict
from src.serializer.interface import DocumentSerializer
from src.train import train
from tests.helpers.run_if import RunIf


@pytest.mark.slow
@pytest.mark.veryslow
def test_predict_cpu(tmp_path, cfg_predict):
    """Run inference on CPU."""

    HydraConfig().set_config(cfg_predict)
    predict(cfg_predict)

    assert path.exists(
        path.join(
            cfg_predict.paths.prediction_save_dir, cfg_predict.dataset_split, "documents.jsonl"
        )
    )
    assert path.exists(path.join(cfg_predict.paths.prediction_save_dir, "config.yaml"))


def test_predict_cpu_fast_dev_run(tmp_path, cfg_predict):
    """Run inference for one batch on CPU."""
    with open_dict(cfg_predict):
        cfg_predict.pipeline.fast_dev_run = True

    HydraConfig().set_config(cfg_predict)
    predict(cfg_predict)

    assert path.exists(
        path.join(
            cfg_predict.paths.prediction_save_dir, cfg_predict.dataset_split, "documents.jsonl"
        )
    )
    assert path.exists(path.join(cfg_predict.paths.prediction_save_dir, "config.yaml"))


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_predict_gpu(tmp_path, cfg_predict):
    """Run inference on GPU."""

    with open_dict(cfg_predict):
        cfg_predict.pipeline.device = 0

    HydraConfig().set_config(cfg_predict)
    predict(cfg_predict)

    assert path.exists(
        path.join(
            cfg_predict.paths.prediction_save_dir, cfg_predict.dataset_split, "documents.jsonl"
        )
    )
    assert path.exists(path.join(cfg_predict.paths.prediction_save_dir, "config.yaml"))


@pytest.mark.slow
@pytest.mark.veryslow
def test_train_predict(tmp_path, cfg_train, cfg_predict):
    """Train for 1 epoch with `train.py` and predict with `predict.py`"""
    assert str(tmp_path) == cfg_train.paths.save_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True
        cfg_train.trainer.limit_train_batches = 10

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert path.exists(cfg_train.paths.model_save_dir)

    with open_dict(cfg_predict):
        cfg_predict.model_name_or_path = cfg_train.paths.model_save_dir

    HydraConfig().set_config(cfg_predict)
    _, object_dict = predict(cfg_predict)

    serializer: DocumentSerializer = object_dict["serializer"]
    documents = serializer.read(
        path=cfg_predict.paths.prediction_save_dir, split=cfg_predict.dataset_split
    )
    predicted_entities = [list(doc.labeled_spans.predictions) for doc in documents]
    num_predicted_entities = sum([len(preds) for preds in predicted_entities])
    assert num_predicted_entities > 0


def test_serialize_only(tmp_path, cfg_predict):
    """Directly serialize the input documents without calling a PyTorch-IE-pipeline."""

    with open_dict(cfg_predict):
        cfg_predict.pipeline = None

    HydraConfig().set_config(cfg_predict)
    predict(cfg_predict)

    assert path.exists(
        path.join(
            cfg_predict.paths.prediction_save_dir, cfg_predict.dataset_split, "documents.jsonl"
        )
    )
    assert path.exists(path.join(cfg_predict.paths.prediction_save_dir, "config.yaml"))
