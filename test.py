import sys
from os.path import dirname, join, realpath

import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(version_base="1.2", config_path="configs/", config_name="test.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.testing_pipeline import test

    # Applies optional utilities
    utils.extras(config)

    # add src to system path to allow its usage with configs
    root_path = realpath(dirname(__file__))
    sys.path.append(join(root_path, "src"))

    # Evaluate model
    return test(config)


if __name__ == "__main__":
    main()
