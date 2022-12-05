import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_config(cfg_train: DictConfig):
    assert cfg_train
    assert cfg_train.dataset
    assert cfg_train.taskmodule
    assert cfg_train.datamodule
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    dataset = hydra.utils.instantiate(cfg_train.dataset)
    taskmodule = hydra.utils.instantiate(cfg_train.taskmodule)
    hydra.utils.instantiate(cfg_train.datamodule, dataset=dataset, taskmodule=taskmodule)
    num_classes_dummy = 3
    hydra.utils.instantiate(cfg_train.model, num_classes=num_classes_dummy)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig):
    assert cfg_eval
    assert cfg_eval.dataset
    assert cfg_eval.taskmodule
    assert cfg_eval.datamodule
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    dataset = hydra.utils.instantiate(cfg_eval.dataset)
    taskmodule = hydra.utils.instantiate(cfg_eval.taskmodule)
    datamodule = hydra.utils.instantiate(
        cfg_eval.datamodule, dataset=dataset, taskmodule=taskmodule
    )
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_predict_config(cfg_predict: DictConfig):
    assert cfg_predict
    assert cfg_predict.dataset
    assert cfg_predict.pipeline
    assert cfg_predict.serializer

    HydraConfig().set_config(cfg_predict)

    dataset = hydra.utils.instantiate(cfg_predict.dataset)
    pipeline = hydra.utils.instantiate(cfg_predict.pipeline)
    serializer = hydra.utils.instantiate(cfg_predict.serializer)
