import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
# to speedup test execution, we use bert-tiny
transformer_model = "prajjwal1/bert-tiny"
overrides = [
    f"model.model_name_or_path={transformer_model}",
    f"taskmodule.tokenizer_name_or_path={transformer_model}",
]


@pytest.mark.skip(reason="this is already covered by tests/test_experiments.py")
@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path):
    """Test running all available experiment configs with fast_dev_run=True."""
    command = [
        startfile,
        "-m",
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
@pytest.mark.veryslow
def test_hydra_sweep(tmp_path):
    """Test default hydra sweep."""
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "+model.learning_rate=0.005,0.01",
        "++trainer.fast_dev_run=true",
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
@pytest.mark.veryslow
def test_hydra_sweep_ddp_sim(tmp_path):
    """Test default hydra sweep with ddp sim."""
    command = (
        [
            startfile,
            "-m",
            "hydra.sweep.dir=" + str(tmp_path),
            "trainer=ddp_sim",
            "trainer.max_epochs=2",
            "+trainer.limit_train_batches=2",
            "+trainer.limit_val_batches=2",
            "+trainer.limit_test_batches=2",
            "+model.learning_rate=0.005,0.01",
        ]
        + overrides
        + ["logger=[]"]
    )
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
@pytest.mark.veryslow
def test_optuna_sweep(tmp_path):
    """Test optuna sweep."""
    command = [
        startfile,
        "-m",
        "hparams_search=conll2003_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=3",
        "hydra.sweeper.sampler.n_startup_trials=2",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(wandb=True, sh=True)
@pytest.mark.slow
def test_optuna_sweep_ddp_sim_wandb(tmp_path):
    """Test optuna sweep with wandb and ddp sim."""
    command = [
        startfile,
        "-m",
        "hparams_search=conll2003_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=3",
        "trainer=ddp_sim",
        "trainer.max_epochs=2",
        "+trainer.limit_train_batches=2",
        "+trainer.limit_val_batches=2",
        "+trainer.limit_test_batches=2",
        "logger=wandb",
    ] + overrides
    run_sh_command(command)
