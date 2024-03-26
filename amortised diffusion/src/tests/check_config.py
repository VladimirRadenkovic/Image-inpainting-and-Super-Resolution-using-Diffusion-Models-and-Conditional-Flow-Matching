import hydra
from omegaconf import DictConfig

from src import constants


# Load hydra config from yaml filses and command line arguments.
@hydra.main(
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name=constants.HYDRA_CONFIG_NAME,
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Load and validate the hydra config."""
    print(cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()