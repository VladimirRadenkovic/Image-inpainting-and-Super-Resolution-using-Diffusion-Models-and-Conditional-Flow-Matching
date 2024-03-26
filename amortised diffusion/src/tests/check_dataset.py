import datetime
import pathlib

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.transforms import Compose
from tqdm.autonotebook import tqdm

from src.constants import HYDRA_CONFIG_NAME, HYDRA_CONFIG_PATH, PROJECT_PATH


def get_full_dataset(cfg: DictConfig) -> None:
    task = hydra.utils.instantiate(cfg.dataset.task)

    pdb_dataset = hydra.utils.instantiate(cfg.dataset.pdb_dataset)
    nma_dataset = hydra.utils.instantiate(cfg.dataset.nma_dataset)

    task_data = task.process(
        processed_dir=cfg.dataset.processed_dir,
        pdb_dataset=pdb_dataset,
        nma_dataset=nma_dataset,
        overwrite=False,
    )

    transforms = hydra.utils.instantiate(cfg.dataset.transforms, add_label={"df": task.data})
    transform = Compose([t for t in transforms.values() if t is not None])

    is_processed = "(processed_path.notnull())"
    processing_failed = "(processed_path.str.contains('failed'))"
    processing_success = f"({is_processed} and not {processing_failed})"

    dataset = hydra.utils.instantiate(
        cfg.dataset.pyg_dataset,
        transform=transform,
        ids=task_data.query(processing_success).id.values,
    )

    return dataset


def check_coord_shape_vs_nma_shape(dataset: "PyGDataset", report_path: pathlib.Path = None) -> None:
    logger.info(f"Checking coord shape vs nma shape for {len(dataset)} proteins")
    for i in tqdm(range(len(dataset))):
        try:
            coords_shape = dataset[i].coords.shape
            eig_vecs_shape = dataset[i].eig_vecs.shape
            assert coords_shape[0] > 0, f"Empty coords: coords_shape={coords_shape}"
            assert (
                coords_shape[0] == eig_vecs_shape[0]
            ), f"Differing seqlen: coords_shape={coords_shape}, eig_vecs_shape={eig_vecs_shape}"

        except Exception as e:
            logger.warning(f"i={i}, id={dataset[i].id}, error: {e}")
            if report_path is not None:
                with open(report_path, "a") as f:
                    f.write(f"i={i}, id={dataset[i].id}, error: {e}\n")


@hydra.main(
    config_path=str(HYDRA_CONFIG_PATH),
    config_name=HYDRA_CONFIG_NAME,
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    dataset = get_full_dataset(cfg)
    report_path = (
        PROJECT_PATH
        / f"tests/check_dataset_{cfg.name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    )
    report_path.parent.mkdir(exist_ok=True, parents=True)
    with open(report_path, "w") as f:
        f.write(str(OmegaConf.to_yaml(cfg)) + "\n" + "=" * 80 + "\n")
    check_coord_shape_vs_nma_shape(dataset, report_path=report_path)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()