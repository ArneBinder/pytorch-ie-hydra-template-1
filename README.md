<div align="center">

# PyTorch-IE-Hydra-Template

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

A clean and scalable template to kickstart your deep learning based information extraction project 🚀⚡🔥<br>
Click on [<kbd>Use this template</kbd>](https://github.com/ChristophAlt/pytorch-ie-hydra-template/generate) to initialize new repository. <br>
This project is heavily inspired by [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).

_Suggestions are always welcome!_

</div>

<br>

## 📌 Introduction

**Why you should use it:**

- Convenient all-in-one technology stack for deep learning based information extraction prototyping - allows you to rapidly iterate over new models, datasets and tasks on different hardware accelerators like CPUs, multi-GPUs or TPUs.
- A collection of best practices for efficient workflow and reproducibility.
- Thoroughly commented - you can use this repo as a reference and educational resource.

**Why you shouldn't use it:**

- Not fitted for data engineering - the template configuration setup is not designed for building data processing pipelines that depend on each other.
- Limits you as much as PyTorch-IE limits you.
- PyTorch-IE and Hydra are still evolving and integrate many libraries, which means sometimes things break - for the list of currently known problems visit [this page](https://github.com/ChristophAlt/pytorch-ie-hydra-template/labels/bug).

<br>

## Main Technologies

[PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) - a lightweight information extraction (IE) technology stack built on top of [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), [Huggingface Datasets](https://github.com/huggingface/datasets) and [Huggingface Hub](https://huggingface.co/) for reproducible high-performance AI research.

[Hydra](https://github.com/facebookresearch/hydra) - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

<br>

## Main Ideas Of This Template

- **Predefined Structure**: clean and scalable so that work can easily be extended and replicated | [#Project Structure](#project-structure)
- **Rapid Experimentation**: thanks to automating pipeline with config files and hydra command line superpowers | [#Your Superpowers](#your-superpowers)
- **Reproducibility**: obtaining similar results is supported in multiple ways | [#Reproducibility](#reproducibility)
- **Little Boilerplate**: so pipeline can be easily modified | [#How It Works](#how-it-works)
- **Main Configuration**: main config file specifies default training configuration | [#Main Project Configuration](#main-project-configuration)
- **Experiment Configurations**: can be composed out of smaller configs and override chosen hyperparameters | [#Experiment Configuration](#experiment-configuration)
- **Workflow**: comes down to 4 simple steps | [#Workflow](#workflow)
- **Experiment Tracking**: many logging frameworks can be easily integrated, like Tensorboard, MLFlow or W&B | [#Experiment Tracking](#experiment-tracking)
- **Logs**: all logs (checkpoints, data from loggers, hparams, etc.) are stored in a convenient folder structure imposed by Hydra | [#Logs](#logs)
- **Hyperparameter Search**: made easier with Hydra built-in plugins like [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) | [#Hyperparameter Search](#hyperparameter-search)
- **Tests**: unit tests and shell/command based tests for speeding up the development | [#Tests](#tests)
- **Best Practices**: a couple of recommended tools, practices and standards for efficient workflow and reproducibility | [#Best Practices](#best-practices)

<br>

## Project Structure

The directory structure of new project looks like this:

```
├── configs                   <- Hydra configuration files
│   ├── callbacks                <- Callbacks configs
│   ├── datamodule               <- Datamodule configs
│   ├── dataset                  <- Dataset configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── taskmodule               <- Taskmodule configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── predict.yaml          <- Main config for inference
│   ├── evaluate.yaml         <- Main config for testing
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│
├── dataset_builders       <- dataset builders
│   ├── hf                   <- Huggingface
│   └── pie                  <- PyTorch-IE
│
├── logs                   <- Logs generated by Hydra and PyTorch Lightning loggers
│   (created on demand)
│
├── models                 <- Per default (see config parameter `save_dir`), trained models and their
│   (created on demand)       respective taskmodules are copied to this location in a format to easily
│                             load them with pytorch_ie.Auto* classes.
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description,
│                             e.g. `1.0-jqp-initial-data-exploration.ipynb`.
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── models                   <- Pytorch-IE models
│   ├── taskmodules              <- Pytorch-IE taskmodules
│   ├── utils                    <- Utility scripts
│   ├── vendor                   <- Third party code that cannot be installed using PIP/Conda
│   │
│   ├── eval.py                  <- Run evaluation
│   ├── predict.py               <- Run inference
│   └── train.py                 <- Run training
│
├── tests                  <- Tests of any kind
│
├── .env.example              <- Template of the file for storing private environment variables
├── .flake8                   <- Configuration of flake8 code analyser tool
├── .gitignore                <- List of files/folders ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── Makefile                  <- Makefile with commands like `make train` or `make test`
├── poetry.lock               <- Lockfile with specific dependency versions automatically managed by Poetry
├── pyproject.toml            <- Project configuration including dependencies, settings for linters and pytest, and building project as a package
├── README.md
└── setup_symlinks.sh         <- Script for automatically creating symlinks to log and model folders in case you want to put them anywhere outside of a project
```

<br>

## 🚀  Quickstart

```bash
# clone project
git clone https://github.com/ChristophAlt/pytorch-ie-hydra-template.git
cd pytorch-ie-hydra-template

# [OPTIONAL] Check the pyproject.toml config file
# - Check if dependency versions fit your needs
# - Uncomment logger you want to use
# - Add your own dependencies manually or with `poetry add abc==1.2.3`
# - Change pytorch version in pyproject.toml to fit your system, see
# https://pytorch.org/get-started/

# Install project and dependencies
poetry install

# Poetry will create a virtual environment for installation by default, unless you have manually disabled it.
# To activate virtual environment, run:
eval $(poetry env activate)


# [OPTIONAL] symlink log directories and the default model directory to
# "$HOME/experiments/my-project" since they can grow a lot (change path to place where you have enough storage)
bash setup_symlinks.sh $HOME/experiments/my-project

# [OPTIONAL] set any environment variables by creating an .env file
# Variables from this file are automatically loaded by train/predict/evaluate_documents scripts via `pyrootutils.setup_root()`
# 1. copy the provided example file:
cp .env.example .env
# 2. edit the .env file for your needs!
```

Template contains example with CONLL2003 Named Entity Recognition.<br>
When running `python train.py` you should see something like this:

<div align="center">

![](https://github.com/ChristophAlt/pytorch-ie-hydra-template/blob/resources/images/terminal1.png)
![](https://github.com/ChristophAlt/pytorch-ie-hydra-template/blob/resources/images/terminal2.png)
![](https://github.com/ChristophAlt/pytorch-ie-hydra-template/blob/resources/images/terminal3.png)

</div>

### ⚡  Your Superpowers

<details>
<summary><b>Override any config parameter from command line</b></summary>

> Hydra allows you to easily overwrite any parameter defined in your config.

```bash
python train.py trainer.max_epochs=20 model.learning_rate=1e-4
```

> You can also add new parameters with `+` sign.

```bash
python train.py +model.new_param="uwu"
```

</details>

<details>
<summary><b>Train on CPU, GPU, multi-GPU and TPU</b></summary>

> PyTorch Lightning makes it easy to train your models on different hardware.

```bash
# train on CPU
python train.py trainer=cpu

# train on 1 GPU
python train.py trainer=gpu

# train on TPU
python train.py +trainer.tpu_cores=8

# train with DDP (Distributed Data Parallel) (4 GPUs)
python train.py trainer=ddp trainer.devices=4

# train with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
python train.py trainer=ddp trainer.devices=4 trainer.num_nodes=2

# simulate DDP on CPU processes
python train.py trainer=ddp_sim trainer.devices=2

# accelerate training on mac
python train.py trainer=mps
```

> **Warning**: Currently there are problems with DDP mode, read [this issue](https://github.com/ashleve/lightning-hydra-template/issues/393) to learn more.

</details>

<details>
<summary><b>Train with mixed precision</b></summary>

```bash
# train with pytorch native automatic mixed precision (AMP)
python train.py trainer=gpu +trainer.precision=16
```

</details>

<!-- deepspeed support still in beta
<details>
<summary><b>Optimize large scale models on multiple GPUs with Deepspeed</b></summary>

```bash
python train.py +trainer.
```

</details>
 -->

<details>
<summary><b>Train model with any logger available in PyTorch Lightning, like Weights&Biases or Tensorboard</b></summary>

> PyTorch Lightning provides convenient integrations with most popular logging frameworks, like Tensorboard, Neptune or simple csv files. Read more [here](#experiment-tracking). Using wandb requires you to [setup account](https://www.wandb.com/) first. After that just complete the config as below.<br> > **Click [here](https://wandb.ai/hobglob/template-dashboard/) to see example wandb dashboard generated with this template.**

```yaml
# set project and entity names in `configs/logger/wandb`
wandb:
  project: "your_project_name"
  entity: "your_wandb_team_name"
```

```bash
# train model with Weights&Biases (link to wandb dashboard should appear in the terminal)
python train.py logger=wandb
```

> **Note**: Lightning provides convenient integrations with most popular logging frameworks. Learn more [here](#experiment-tracking).

> **Note**: Using wandb requires you to [setup account](https://www.wandb.com/) first. After that just complete the config as below.

> **Note**: Click [here](https://wandb.ai/hobglob/template-dashboard/) to see example wandb dashboard generated with this template.

</details>

<details>
<summary><b>Train model with chosen experiment config</b></summary>

```bash
python train.py experiment=example
```

> **Note**: Experiment configs are placed in [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Attach some callbacks to run</b></summary>

```bash
python train.py callbacks=default
```

> **Note**: Callbacks can be used for things such as as model checkpointing, early stopping and [many more](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks).

> **Note**: Callbacks configs are placed in [configs/callbacks/](configs/callbacks/).

</details>

<details>
<summary><b>Use different tricks available in Pytorch Lightning</b></summary>

```yaml
# gradient clipping may be enabled to avoid exploding gradients
python train.py +trainer.gradient_clip_val=0.5

# run validation loop 4 times during a training epoch
python train.py +trainer.val_check_interval=0.25

# accumulate gradients
python train.py +trainer.accumulate_grad_batches=10

# terminate training after 12 hours
python train.py +trainer.max_time="00:12:00:00"
```

> **Note**: PyTorch Lightning provides about [40+ useful trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).

</details>

<details>
<summary><b>Easily debug</b></summary>

```bash
# runs 1 epoch in default debugging mode
# changes logging directory to `logs/debugs/...`
# sets level of all command line loggers to 'DEBUG'
# enforces debug-friendly configuration
python train.py debug=default

# run 1 train, val and test loop, using only 1 batch
python train.py debug=fdr

# print execution time profiling
python train.py debug=profiler

# try overfitting to 1 batch
python train.py debug=overfit

# raise exception if there are any numerical anomalies in tensors, like NaN or +/-inf
python train.py +trainer.detect_anomaly=true

# log second gradient norm of the model
python train.py +trainer.track_grad_norm=2

# use only 20% of the data
python train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2
```

> **Note**: Visit [configs/debug/](configs/debug/) for different debugging configs.

</details>

<details>
<summary><b>Resume training from checkpoint</b></summary>

```yaml
python train.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

> **Note**: Currently loading ckpt doesn't resume logger experiment, but it will be supported in future Lightning release.

</details>

<details>
<summary><b>Evaluate checkpoint on test dataset</b></summary>

```yaml
python eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

</details>

<details>
<summary><b>Create a sweep over hyperparameters</b></summary>

```bash
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python train.py -m datamodule.batch_size=32,64,128 model.learning_rate=0.001,0.0005
```

> **Note**: Hydra composes configs lazily at job launch time. If you change code or configs after launching a job/sweep, the final composed configs might be impacted.

</details>

<details>
<summary><b>Create a sweep over hyperparameters with Optuna</b></summary>

> Using [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) plugin doesn't require you to code any boilerplate into your pipeline, everything is defined in a [single config file](configs/hparams_search/conll2003_optuna.yaml)!

```bash
# this will run hyperparameter search defined in `configs/hparams_search/conll2003_optuna.yaml`
# over chosen experiment config
python train.py -m hparams_search=conll2003_optuna experiment=conll2003
```

> ⚠️**Warning**: Optuna sweeps are not failure-resistant (if one job crashes then the whole sweep crashes).

</details>

<details>
<summary><b>Execute all experiments from folder</b></summary>

```bash
python train.py -m 'experiment=glob(*)'
```

> **Note**: Hydra provides special syntax for controlling behavior of multiruns. Learn more [here](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run). The command above executes all experiments from [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Execute sweep on a remote AWS cluster</b></summary>

> **Note**: This should be achievable with simple config using [Ray AWS launcher for Hydra](https://hydra.cc/docs/next/plugins/ray_launcher). Example is not implemented in this template.

</details>

<!-- <details>
<summary><b>Execute sweep on a SLURM cluster</b></summary>

> This should be achievable with either [the right lightning trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster.html?highlight=SLURM#slurm-managed-cluster) or simple config using [Submitit launcher for Hydra](https://hydra.cc/docs/plugins/submitit_launcher). Example is not yet implemented in this template.

</details> -->

<details>
<summary><b>Use Hydra tab completion</b></summary>

> **Note**: Hydra allows you to autocomplete config argument overrides in shell as you write them, by pressing `tab` key. Read the [docs](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion).

</details>

<details>
<summary><b>Apply pre-commit hooks</b></summary>

```bash
pre-commit run -a
```

> **Note**: Apply pre-commit hooks to do things like auto-formatting code and configs, performing code analysis or removing output from jupyter notebooks. See [# Best Practices](#best-practices) for more.

</details>

<details>
<summary><b>Run tests</b></summary>

```bash
# run all tests
pytest

# run tests from specific file
pytest tests/test_train.py

# run all tests except the ones marked as slow
pytest -m "not slow"

# we are using pytest-xdist to run tests in parallel
# by default tests will create as many workers as there are cores available
# run tests with only 4 workers
pytest -n=4

# disable this feature completely and run tests as usual
pytest -n=0
```

</details>

<details>
<summary><b>Use tags</b></summary>

Each experiment should be tagged in order to easily filter them across files or in logger UI:

```bash
python train.py tags=["conll2003","experiment_X"]
```

If no tags are provided, you will be asked to input them from command line:

```bash
>>> python train.py tags=[]
[2022-07-11 15:40:09,358][src.utils.utils][INFO] - Enforcing tags! <cfg.extras.enforce_tags=True>
[2022-07-11 15:40:09,359][src.utils.rich_utils][WARNING] - No tags provided in config. Prompting user to input tags...
Enter a list of comma separated tags (dev):
```

If no tags are provided for multirun, an error will be raised:

```bash
>>> python train.py -m +x=1,2,3 tags=[]
ValueError: Specify tags before launching a multirun!
```

> **Note**: Appending lists from command line is currently not supported in hydra :(

</details>

<br>

## ❤️  Contributions

Before making an issue, please verify that:

- The problem still exists on the current `main` branch.
- Your python dependencies are updated to recent versions.

Suggestions for improvements are always welcome!

<br>

## How To Get Started

- First, you should probably get familiar with [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie)
- Next, go through [Hydra quick start guide](https://hydra.cc/docs/intro/) and [basic Hydra tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/)

<br>

## How It Works

All PyTorch-IE modules are dynamically instantiated from module paths specified in config. Example model config:

```yaml
_target_: pytorch_ie.models.transformer_token_classification.TransformerTokenClassificationModel

model_name_or_path: bert-base-uncased
learning_rate: 0.005
```

Using this config we can instantiate the object with the following line:

```python
model = hydra.utils.instantiate(config.model)
```

This allows you to easily iterate over new models! Every time you create a new one, just specify its module path and parameters in appropriate config file. <br>

Switch between models and datasets with command line arguments:

```bash
python train.py model=transformer_token_classification
```

The whole pipeline managing the instantiation logic is placed in [src/train.py](src/train.py).

<br>

## Main Configuration

Location: [configs/train.yaml](configs/train.yaml) <br>
Main project config contains default training configuration.<br>
It determines how config is composed when simply executing command `python train.py`.<br>

<details>
<summary><b>Show main project config</b></summary>

```yaml
# @package _global_

# specify here default configuration
# order of defaults determines the order in which configs override each other
defaults:
  - _self_
  - dataset: conll2003.yaml
  - datamodule: default.yaml
  - taskmodule: transformer_token_classification.yaml
  - model: transformer_token_classification.yaml
  - callbacks: default.yaml
  - logger: null # set logger here or use command line (e.g. `python train.py logger=tensorboard`)
  - trainer: default.yaml
  - paths: default.yaml
  - extras: default.yaml
  - hydra: default.yaml

  # experiment configs allow for version control of specific hyperparameters
  # e.g. best hyperparameters for given model and taskmodule
  - experiment: null

  # config for hyperparameter optimization
  - hparams_search: null

  # optional local config for machine/user specific settings
  # it's optional since it doesn't need to exist and is excluded from version control
  - optional local: default.yaml

  # debugging config (enable through command line, e.g. `python train.py debug=default)
  - debug: null

# task name, determines output directory path
pipeline_type: "training"

# default name for the experiment, determines output directory path
# (you can overwrite this in experiment configs)
name: "default"

# tags to help you identify your experiments
# you can overwrite this in experiment configs
# overwrite from command line with `python train.py tags="[first_tag, second_tag]"`
# appending lists from command line is currently not supported :(
# https://github.com/facebookresearch/hydra/issues/1547
tags: ["dev"]

# set False to skip model training
train: True

# evaluate on test set, using best model weights achieved during training
# lightning chooses best weights based on the metric specified in checkpoint callback
test: False

# seed for random number generators in pytorch, numpy and python.random
seed: null

# simply provide checkpoint path to resume training
ckpt_path: null

# push the model and taskmodule to the huggingface model hub when training has finished
push_to_hub: False

# where to save the trained model and taskmodule
save_dir: models/${name}/${now:%Y-%m-%d_%H-%M-%S}
```

</details>

<br>

## Experiment Configuration

Location: [configs/experiment](configs/experiment)<br>
Experiment configs allow you to overwrite parameters from main project configuration.<br>
For example, you can use them to version control best hyperparameters for each combination of model and dataset.

<details>
<summary><b>Show example experiment config</b></summary>

```yaml
# @package _global_

# to execute this experiment run:
# python train.py experiment=conll2003

defaults:
  - override /dataset: conll2003.yaml
  - override /datamodule: default.yaml
  - override /taskmodule: transformer_token_classification.yaml
  - override /model: transformer_token_classification.yaml
  - override /callbacks: default.yaml
  - override /logger: aim.yaml
  - override /trainer: default.yaml

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

# name of the run determines folder name in logs
name: "conll2003/transformer_token_classification"

tags: ["dataset=conll2003", "model=transformer_token_classification"]

seed: 12345

trainer:
  min_epochs: 5
  max_epochs: 20
  # gradient_clip_val: 0.5

datamodule:
  batch_size: 32
```

</details>

<br>

## Local Configuration

Location: [configs/local](configs/local) <br>
Some configurations are user/machine/installation specific (e.g. configuration of local cluster, or harddrive paths on a specific machine). For such scenarios, a file `configs/local/default.yaml` can be created which is automatically loaded but not tracked by Git.

<details>
<summary><b>Show example local Slurm cluster config</b></summary>

```yaml
# @package _global_

defaults:
  - override /hydra/launcher@_here_: submitit_slurm

data_dir: /mnt/scratch/data/

hydra:
  launcher:
    timeout_min: 1440
    gpus_per_task: 1
    gres: gpu:1
  job:
    env_set:
      MY_VAR: /home/user/my/system/path
      MY_KEY: asdgjhawi8y23ihsghsueity23ihwd
```

</details>

<br>

## Workflow

Before creating your own setup, have a look into the
[Pytorch-IE documentation](https://github.com/ChristophAlt/pytorch-ie#-concepts--architecture) to make yourself
familiar with the Pytorch-IE core concepts like the `document`, `model`, and `taskmodule`.

1. Write your PyTorch-IE dataset loader (see [dataset_builders/pie/conll2003/conll2003.py](dataset_builders/pie/conll2003/conll2003.py) for an example) or try out one of the PIE datasets hosted at [huggingface.co/pie](https://huggingface.co/pie).
2. Write your PyTorch-IE model (see [src/models/transformer_token_classification.py](src/models/transformer_token_classification.py) for an example) or use one of the implementations from [pytorch-ie](https://github.com/ChristophAlt/pytorch-ie) or [pie-modules](https://github.com/ArneBinder/pie-modules).
3. Write your PyTorch-IE taskmodule (see [src/taskmodules/transformer_token_classification.py](src/taskmodules/transformer_token_classification.py) for example) or use one of the implementations from [pytorch-ie](https://github.com/ChristophAlt/pytorch-ie) or [pie-modules](https://github.com/ArneBinder/pie-modules).
4. Write your experiment config, containing paths to your model, taskmodule and dataset (see [configs/experiment/conll2003.yaml](configs/experiment/conll2003.yaml) for example). You may need to also write configs for your model, taskmodule and dataset, if you do not want to use the default ones.
5. If necessary, define `additional_model_kwargs` for your model class in the [train.py](src/train.py) (see line with `# NOTE: MODIFY THE additional_model_kwargs IF YOUR MODEL REQUIRES ...`").
6. Execute a dev run for your setup to ensure that everything works as expected (assuming that `configs/experiments/experiment_name.yaml` is your experiment config file): `python src/train.py experiment=experiment_name +trainer.fast_dev_run=true`
7. Run training with chosen experiment config on the GPU: `python src/train.py experiment=experiment_name trainer=gpu`

<br>

## Logs

**Hydra creates new working directory for every executed run.** By default, logs have the following structure:

```
├── logs
│   ├── pipeline_type                     # Folder for the logs generated by type of pipeline, i.e. training, evaluation, or prediction
│   │   ├── runs                          # Folder for single runs
│   │   │   ├── experiment_name             # Experiment name
│   │   │   │   ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the run
│   │   │   │   │   ├── .hydra                  # Hydra logs
│   │   │   │   │   ├── csv                     # Csv logs
│   │   │   │   │   ├── wandb                   # Weights&Biases logs
│   │   │   │   │   ├── checkpoints             # Training checkpoints
│   │   │   │   │   └── ...                     # Any other thing saved during training
│   │   │   │   └── ...
│   │   │   └── ...
│   │   │
│   │   └── multiruns                     # Folder for multiruns
│   │       ├── experiment_name             # Experiment name
│   │       │   ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the multirun
│   │       │   │   ├──1                        # Multirun job number
│   │       │   │   ├──2
│   │       │   │   └── ...
│   │       │   └── ...
│   │       └── ...
│   │
│   └── debugs                            # Logs generated when debugging config is attached
│       └── ...
```

You can change this structure by modifying paths in [hydra configuration](configs/log_dir).

<br>

## Experiment Tracking

PyTorch-IE is based on PyTorch Lightning which supports many popular logging frameworks:<br>
\*\*[Weights&Biases](https://www.wandb.com/) · [Neptune](https://neptune.ai/) · [Comet](https://www.comet.ml/) · [MLFlow](https://mlflow.org) · [Tensorboard](https://www.tensorflow.org/tensorboard/) · [Aim](https://aimstack.io/) \*\*

These tools help you keep track of hyperparameters and output metrics and allow you to compare and visualize results. To use one of them simply complete its configuration in [configs/logger](configs/logger) and run:

```bash
python train.py logger=logger_name
```

You can use many of them at once (see [configs/logger/many_loggers.yaml](configs/logger/many_loggers.yaml) for example).

You can also write your own logger.

Lightning provides convenient method for logging custom metrics from inside LightningModule. Read the docs [here](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#automatic-logging) or take a look at [TransformerTokenClassification example](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/models/transformer_token_classification.py).

<br>

## Tests

Template comes with generic tests implemented with `pytest`.

```bash
# run all tests
pytest

# run tests from specific file
pytest tests/test_train.py

# run all tests except the ones marked as slow
pytest -m "not slow"

# we are using pytest-xdist to run tests in parallel
# by default tests will create as many workers as there are cores available
# run tests with only 4 workers
pytest -n=4

# disable this feature completely and run tests as usual
pytest -n=0
```

Most of the implemented tests don't check for any specific output - they exist to simply verify that executing some commands doesn't end up in throwing exceptions. You can execute them once in a while to speed up the development.

Currently, the tests cover cases like:

- running 1 train, val and test step
- running 1 epoch on 1% of data, saving ckpt and resuming for the second epoch
- running 2 epochs on 1% of data, with DDP simulated on CPU

And many others. You should be able to modify them easily for your use case.

There is also `@RunIf` decorator implemented, that allows you to run tests only if certain conditions are met, e.g. GPU is available or system is not windows. See the [examples](tests/test_train.py).

<br>

## Hyperparameter Search

You can define hyperparameter search by adding new config file to [configs/hparams_search](configs/hparams_search).

<details>
<summary><b>Show example hyperparameter search config</b></summary>

```yaml
# @package _global_

# example hyperparameter optimization of some experiment with Optuna:
# python train.py -m hparams_search=conll2003_optuna experiment=conll2003

defaults:
  - override /hydra/sweeper: optuna

# choose metric which will be optimized by Optuna
# make sure this is the correct name of some metric logged in lightning module!
optimized_metric: "val/f1"

# here we define Optuna hyperparameter search
# it optimizes for value returned from function with @hydra.main decorator
# docs: https://hydra.cc/docs/next/plugins/optuna_sweeper
hydra:
  mode: "MULTIRUN" # set hydra to multirun by default if this config is attached

  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    # storage URL to persist optimization results
    # for example, you can use SQLite if you set 'sqlite:///example.db'
    storage: null

    # name of the study to persist optimization results
    study_name: null

    # number of parallel workers
    n_jobs: 1

    # 'minimize' or 'maximize' the objective
    direction: maximize

    # total number of runs that will be executed
    n_trials: 20

    # choose Optuna hyperparameter sampler
    # you can choose bayesian sampler (tpe), random search (without optimization), grid sampler, and others
    # docs: https://optuna.readthedocs.io/en/stable/reference/samplers.html
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 1234
      n_startup_trials: 10 # number of random sampling runs before optimization starts

    # define range of hyperparameters
    # More information here : https://hydra.cc/docs/plugins/optuna_sweeper/#search-space-configuration
    params:
      datamodule.batch_size: choice(32,64,128)
      model.learning_rate: interval(0.0001, 0.2)

# This is a dummy value necessary to allow overwriting it in the sweep.
model:
  learning_rate: 0.00001
```

</details>

Next, you can execute it with: `python train.py -m hparams_search=conll2003_optuna`

Using this approach doesn't require adding any boilerplate to code, everything is defined in a single config file. The only necessary thing is to return the optimized metric value from the launch file.

You can use different optimization frameworks integrated with Hydra, like [Optuna, Ax or Nevergrad](https://hydra.cc/docs/plugins/optuna_sweeper/).

The `optimization_results.yaml` will be available under `logs/pipeline_type/multirun` folder.

This approach doesn't support advanced techniques like prunning - for more sophisticated search, you should probably write a dedicated optimization task (without multirun feature).

<br>

## Continuous Integration

Template comes with CI workflows implemented in Github Actions:

- `.github/actions/init-env`: Set up poetry environment for workflow
- `.github/workflows/code_quality_and_tests.yaml`:
  - `pre-commit`: code quality checks, see [.pre-commit-config.yaml](.pre-commit-config.yaml) for configured entries
  - `pytest`:
    - running tests that are not marked as "slow" in PRs
    - running all tests on push to main branch

> **Note**: You need to enable the GitHub Actions from the settings in your repository.

<br>

## Distributed Training

Lightning supports multiple ways of doing distributed training. The most common one is DDP, which spawns separate process for each GPU and averages gradients between them. To learn about other approaches read the [lightning docs](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html).

You can run DDP on mnist example with 4 GPUs like this:

```bash
python train.py trainer=ddp
```

> **Note**: When using DDP you have to be careful how you write your models - read the [docs](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html).

<br>

## Accessing Datamodule Attributes In Model

The simplest way is to pass datamodule attribute directly to model on initialization:

```python
# ./src/train.py
datamodule = hydra.utils.instantiate(config.datamodule)
model = hydra.utils.instantiate(config.model, some_param=datamodule.some_param)
```

> **Note**: Not a very robust solution, since it assumes all your datamodules have `some_param` attribute available.

Similarly, you can pass a whole datamodule config as an init parameter:

```python
# ./src/train.py
model = hydra.utils.instantiate(config.model, dm_conf=config.datamodule, _recursive_=False)
```

You can also pass a datamodule config parameter to your model through variable interpolation:

```yaml
# ./configs/model/my_model.yaml
_target_: src.models.my_module.MyLitModule
lr: 0.01
some_param: ${datamodule.some_param}
```

Another approach is to access datamodule in LightningModule directly through Trainer:

```python
# ./src/models/mnist_module.py
def on_train_start(self):
  self.some_param = self.trainer.datamodule.some_param
```

> **Note**: This only works after the training starts since otherwise trainer won't be yet available in LightningModule.

<br>

## Inference

The following code is an example of loading model from checkpoint and running predictions.<br>

<details>
<summary><b>Show example</b></summary>

```python
from dataclasses import dataclass

from pytorch_ie import PyTorchIEPipeline
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationLayer, annotation_field
from pytorch_ie.documents import TextDocument

@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


def predict():
   """
   Example of inference with trained model.
   It loads pretrained NER model. Then it
   creates an example document (PyTorch-IE Document) and predicts
   entities from the text in the document.
   """
   document = ExampleDocument(
       "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
   )

   # model path can be set to a location at huggingface as shown below or local path to the training result serialized to out_path
   ner_pipeline = PyTorchIEPipeline.from_pretrained("pie/example-ner-spanclf-conll03", device=-1, num_workers=0)

   ner_pipeline(document)

   for entity in document.entities.predictions:
       print(f"{entity} -> {entity.label}")

if __name__ == "__main__":
    predict()

# Result:
# IndieBio -> ORG
# Po Bronson -> PER
# SOSV -> ORG
```

</details>

<br>

### Reproducibility

What provides reproducibility:

- Hydra manages your configs
- Hydra manages your logging paths and makes every executed run store its hyperparameters and config overrides in a separate file in logs
- Single seed for random number generators in pytorch, numpy and python.random
- LightningDataModule allows you to encapsulate data split, transformations and default parameters in a single, clean abstraction
- LightningModule separates your research code from engineering code in a clean way
- Experiment tracking frameworks take care of logging metrics and hparams, some can also store results and artifacts in cloud
- Pytorch Lightning takes care of creating training checkpoints
- Example callbacks for wandb show how you can save and upload a snapshot of codebase every time the run is executed, as well as upload ckpts and track model gradients

<!--
You can load the config of previous run using:

```bash
python train.py --config-path /logs/runs/.../.hydra/ --config-name config.yaml
```

The `config.yaml` from `.hydra` folder contains all overridden parameters and sections. This approach however is not officially supported by Hydra and doesn't override the `hydra/` part of the config, meaning logging paths will revert to default!
 -->

<br>

## Best Practices

<details>
<summary><b>Use Miniconda for GPU environments</b></summary>

Use miniconda for your python environments (it's usually unnecessary to install full anaconda environment, miniconda should be enough).
It makes it easier to install some dependencies, like cudatoolkit for GPU support. It also allows you to access your environments globally.

Example installation:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Create new conda environment:

```bash
conda create -n myenv python=3.8
conda activate myenv
```

</details>

<details>
<summary><b>Use automatic code formatting</b></summary>

Use pre-commit hooks to standardize code formatting of your project and save mental energy.<br>
Simply install pre-commit package with:

```bash
pip install pre-commit
```

Next, install hooks from [.pre-commit-config.yaml](.pre-commit-config.yaml):

```bash
pre-commit install
```

After that your code will be automatically reformatted on every new commit.

Currently template contains configurations of:

- **black** (python code formatting)
- **isort** (python import sorting)
- **pyupgrade** (upgrading python syntax to newer version)
- **docformatter** (python docstring formatting)
- **flake8** (python pep8 code analysis)
- **prettier** (yaml formatting)
- **nbstripout** (clearing output from jupyter notebooks)
- **bandit** (python security linter)
- **mdformat** (markdown formatting)
- **codespell** (word spellling linter)

To reformat all files in the project use command:

```bash
pre-commit run -a
```

</details>

<details>
<summary><b>Set private environment variables in .env file</b></summary>

System specific variables (e.g. absolute paths to datasets) should not be under version control or it will result in conflict between different users. Your private keys also shouldn't be versioned since you don't want them to be leaked.<br>

Template contains `.env.example` file, which serves as an example. Create a new file called `.env` (this name is excluded from version control in .gitignore).
You should use it for storing environment variables like this:

```
MY_VAR=/home/user/my_system_path
```

All variables from `.env` are loaded in `train.py` automatically.

Hydra allows you to reference any env variable in `.yaml` configs like this:

```yaml
path_to_data: ${oc.env:MY_VAR}
```

</details>

<details>
<summary><b>Name metrics using '/' character</b></summary>

Depending on which logger you're using, it's often useful to define metric name with `/` character:

```python
self.log("train/loss", loss)
```

This way loggers will treat your metrics as belonging to different sections, which helps to get them organised in UI.

</details>

<details>
<summary><b>Use torchmetrics</b></summary>

Use official [torchmetrics](https://github.com/PytorchLightning/metrics) library to ensure proper calculation of metrics. This is especially important for multi-GPU training!

For example, instead of calculating accuracy by yourself, you should use the provided `Accuracy` class like this:

```python
from torchmetrics.classification.accuracy import Accuracy


class LitModel(LightningModule):
    def __init__(self)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        ...
        acc = self.train_acc(predictions, targets)
        self.log("train/acc", acc)
        ...

    def validation_step(self, batch, batch_idx):
        ...
        acc = self.val_acc(predictions, targets)
        self.log("val/acc", acc)
        ...
```

Make sure to use different metric instance for each step to ensure proper value reduction over all GPU processes.

Torchmetrics provides metrics for most use cases, like F1 score or confusion matrix. Read [documentation](https://torchmetrics.readthedocs.io/en/latest/#more-reading) for more.

</details>

<details>
<summary><b>Follow PyTorch Lightning style guide</b></summary>

The style guide is available [here](https://pytorch-lightning.readthedocs.io/en/latest/starter/style_guide.html).<br>

1. Be explicit in your init. Try to define all the relevant defaults so that the user doesn’t have to guess. Provide type hints. This way your module is reusable across projects!

   ```python
   class LitModel(LightningModule):
       def __init__(self, layer_size: int = 256, lr: float = 0.001):
   ```

2. Preserve the recommended method order.

   ```python
   class LitModel(LightningModule):

       def __init__():
           ...

       def forward():
           ...

       def training_step():
           ...

       def training_step_end():
           ...

       def training_epoch_end():
           ...

       def validation_step():
           ...

       def validation_step_end():
           ...

       def validation_epoch_end():
           ...

       def test_step():
           ...

       def test_step_end():
           ...

       def test_epoch_end():
           ...

       def configure_optimizers():
           ...

       def any_extra_hook():
           ...
   ```

</details>

<details>
<summary><b>Version control your data and models with DVC</b></summary>

Use [DVC](https://dvc.org) to version control big files, like your data or trained ML models.<br>
To initialize the dvc repository:

```bash
dvc init
```

To start tracking a file or directory, use `dvc add`:

```bash
dvc add data/MNIST
```

DVC stores information about the added file (or a directory) in a special .dvc file named data/MNIST.dvc, a small text file with a human-readable format. This file can be easily versioned like source code with Git, as a placeholder for the original data:

```bash
git add data/MNIST.dvc data/.gitignore
git commit -m "Add raw data"
```

</details>

<details>
<summary><b>Support installing project as a package</b></summary>

It allows other people to easily use your modules in their own projects.
Change name of the `src` folder to your project name and complete the `setup.py` file.

Now your project can be installed from local files:

```bash
pip install -e .
```

Or directly from git repository:

```bash
pip install git+git://github.com/YourGithubName/your-repo-name.git --upgrade
```

So any file can be easily imported into any other file like so:

```python
from project_name.models.mnist_module import MNISTLitModule
from project_name.datamodules.mnist_datamodule import MNISTDataModule
```

</details>

<details>
<summary><b>Keep local configs out of code versioning</b></summary>

Some configurations are user/machine/installation specific (e.g. configuration of local cluster, or harddrive paths on a specific machine). For such scenarios, a file [configs/local/default.yaml](configs/local/) can be created which is automatically loaded but not tracked by Git.

Example SLURM cluster config:

```yaml
# @package _global_

defaults:
  - override /hydra/launcher@_here_: submitit_slurm

data_dir: /mnt/scratch/data/

hydra:
  launcher:
    timeout_min: 1440
    gpus_per_task: 1
    gres: gpu:1
  job:
    env_set:
      MY_VAR: /home/user/my/system/path
      MY_KEY: asdgjhawi8y23ihsghsueity23ihwd
```

</details>

<br>

## Other Repositories

<details>
<summary><b>Inspirations</b></summary>

This template was inspired by:
[PyTorchLightning/deep-learninig-project-template](https://github.com/PyTorchLightning/deep-learning-project-template),
[drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science),
[tchaton/lightning-hydra-seed](https://github.com/tchaton/lightning-hydra-seed),
[Erlemar/pytorch_tempest](https://github.com/Erlemar/pytorch_tempest),
[lucmos/nn-template](https://github.com/lucmos/nn-template).

</details>

<details>
<summary><b>Useful repositories</b></summary>

- [pytorch/hydra-torch](https://github.com/pytorch/hydra-torch) - resources for configuring PyTorch classes with Hydra,
- [romesco/hydra-lightning](https://github.com/romesco/hydra-lightning) - resources for configuring PyTorch Lightning classes with Hydra
- [lucmos/nn-template](https://github.com/lucmos/nn-template) - similar template
- [PyTorchLightning/lightning-transformers](https://github.com/PyTorchLightning/lightning-transformers) - official Lightning Transformers repo built with Hydra

</details>

<!-- ## :star:&nbsp; Stargazers Over Time
[![Stargazers over time](https://starchart.cc/ashleve/lightning-hydra-template.svg)](https://starchart.cc/ashleve/lightning-hydra-template) -->

<br>

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

<br>
<br>
<br>
<br>

**DELETE EVERYTHING ABOVE FOR YOUR PROJECT**

______________________________________________________________________

<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ChristophAlt/pytorch-ie-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-PyTorch--IE--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## 📌 Description

What it does

## 🚀 Quickstart

### Environment Setup

```bash
# clone project
git clone https://github.com/your-github-name/your-project-name.git
cd your-project-name

# Install project and dependencies
poetry install

# Poetry will create a virtual environment for installation by default, unless you have manually disabled it.
# To activate virtual environment, run:
eval $(poetry env activate)

# [OPTIONAL] symlink log directories and the default model directory to
# "$HOME/experiments/your-project-name" since they can grow a lot
bash setup_symlinks.sh $HOME/experiments/your-project-name

# [OPTIONAL] set any environment variables by creating an .env file
# Variables from this file are automatically loaded by train/predict/evaluate_documents scripts via `pyrootutils.setup_root()`
# 1. copy the provided example file:
cp .env.example .env
# 2. edit the .env file for your needs!
```

### Model Training

**Have a look into the [train.yaml](configs/train.yaml) config to see all available options.**

Train model with default configuration

```bash
# train on CPU
python src/train.py

# train on GPU
python src/train.py trainer=gpu
```

Execute a fast development run (train for two steps)

```bash
python src/train.py +trainer.fast_dev_run=true
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=conll2003
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```

Start multiple runs at once (multirun):

```bash
python src/train.py seed=42,43 --multirun
```

Notes:

- this will execute two experiments (one after the other), one for each seed
- the results will be aggregated and stored in `logs/multirun/`, see the last logging output for the exact path

### Model evaluation

This will evaluate the model on the test set of the chosen dataset using the *metrics implemented within the model*.
See [config/dataset/](configs/dataset/) for available datasets.

**Have a look into the [evaluate.yaml](configs/evaluate.yaml) config to see all available options.**

```bash
python src/train.py --config-name=evaluate dataset=conll2003 model_name_or_path=pie/example-ner-spanclf-conll03
```

Notes:

- add the command line parameter `trainer=gpu` to run on GPU

### Inference

This will run inference on the given dataset and split. See [config/dataset/](configs/dataset/) for available datasets.
The result documents including the predicted annotations will be stored in the `predictions/` directory (exact
location will be printed to the console).

**Have a look into the [predict.yaml](configs/predict.yaml) config to see all available options.**

```bash
python src/predict.py dataset=conll2003 model_name_or_path=pie/example-ner-spanclf-conll03
```

Notes:

- add the command line parameter `+pipeline.device=0` to run the inference on GPU 0

### Evaluate Serialized Documents

This will evaluate serialized documents including predicted annotations (see [Inference](#inference)) using a
*document metric*. See [config/metric/](configs/metric/) for available metrics.

**Have a look into the [evaluate_documents.yaml](configs/evaluate_documents.yaml) config to see all available options**

```bash
python src/evaluate_documents.py metric=f1 metric.layer=entities +dataset.data_dir=PATH/TO/DIR/WITH/SPLITS
```

Note: By default, this utilizes the dataset provided by the
[from_serialized_documents](configs/dataset/from_serialized_documents.yaml) configuration. This configuration is
designed to facilitate the loading of serialized documents, as generated during the [Inference](#inference) step. It
requires to set the parameter `data_dir`. If you want to use a different dataset,
you can override the `dataset` parameter as usual with any existing dataset config, e.g `dataset=conll2003`. But
calculating the F1 score on the bare `conll2003` dataset does not make much sense, because it does not contain any
predictions. However, it could be used with statistical metrics such as
[count_text_tokens](configs/metric/count_text_tokens.yaml) or
[count_entity_labels](configs/metric/count_entity_labels.yaml).

## Development

```bash
# run code formatting, code analysis, static type checking, and more (see .pre-commit-config.yaml)
pre-commit run -a
```

```bash
# or you can install the pre-commit git hook (this will automatically run checks every time you create a commit) by running
pre-commit install
```

```bash
# run tests
pytest -m "not slow" --cov --cov-report term-missing
```
