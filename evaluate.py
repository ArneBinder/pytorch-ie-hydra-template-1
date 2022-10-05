import sys
from os.path import dirname, join, realpath

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

# register replace resolver (used to replace "/" with "-" in names to use them as e.g. wandb project names)
OmegaConf.register_new_resolver("replace", lambda s, x, y: s.replace(x, y))


@hydra.main(version_base="1.2", config_path="configs/", config_name="evaluate.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.evaluation_pipeline import evaluate

    # Applies optional utilities
    utils.extras(config)

    # add src to system path to allow its usage with configs
    root_path = realpath(dirname(__file__))
    sys.path.append(join(root_path, "src"))

    # Evaluate model
    return evaluate(config)


if __name__ == "__main__":
    main()
