# Convenient setup for jupyter notebook explorations
# Use via inserting:
#   `from src.utils.notebook import *`
# as the last import in your jupyter notebook imports cell.

import os
import pathlib

import dotenv
import hydra
from loguru import logger

from src.constants import HYDRA_CONFIG_PATH

dotenv.load_dotenv(override=True)
try:
    # Convenient for debugging
    import lovely_tensors as lt
    from icecream import ic  # noqa

    lt.monkey_patch()
except:
    pass


# Convenient aliases:
instantiate = hydra.utils.instantiate


def init_hydra_singleton(
    path: os.PathLike = HYDRA_CONFIG_PATH, reload: bool = False, version_base="1.2"
) -> None:
    # See: https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
    if reload:
        clear_hydra_singleton()
    try:
        path = pathlib.Path(path)
        # Note: hydra needs to be initialised with a relative path. Since the hydra
        #  singleton is first created here, it needs to be created relative to this
        #  file. The `rel_path` below takes care of that.
        rel_path = os.path.relpath(path, start=pathlib.Path(__file__).parent)
        hydra.initialize(rel_path, version_base=version_base)
        logger.info("Hydra initialised at %s." % path.absolute())
    except ValueError:
        logger.info("Hydra already initialised.")


def clear_hydra_singleton() -> None:
    if hydra.core.global_hydra.GlobalHydra not in hydra.core.singleton.Singleton._instances:
        return
    hydra_singleton = hydra.core.singleton.Singleton._instances[hydra.core.global_hydra.GlobalHydra]
    hydra_singleton.clear()
    logger.info("Hydra singleton cleared and ready to re-initialise.")