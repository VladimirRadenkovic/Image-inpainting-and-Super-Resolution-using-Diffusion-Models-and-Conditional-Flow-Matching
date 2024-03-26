import torch
from functools import cached_property
import pandas as pd
from typing import Callable, Any, Union, Set, Optional
import numpy as np

from loguru import logger
from tqdm.autonotebook import tqdm


class PandasDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable[[dict], Any] = None,
        id_col: Optional[str] = "id",
    ):
        self.df = df
        self.id_col = id_col
        if id_col is not None:
            assert id_col in self.df.columns, f"ID column {id_col} not in dataframe."
            id_data = self.df[id_col]
        else:
            id_data = self.df.index
        assert id_data.is_unique, f"ID column {id_col} is not unique."
        assert id_data.notnull().all(), f"ID column {id_col} has null values."

        self._ids = id_data.values
        self._id_to_idx = {id: idx for idx, id in enumerate(self._ids)}
        self.transform = transform

    @cached_property
    def ids(self) -> Set[str]:
        return set(self._ids)

    def __len__(self) -> int:
        return len(self.df)

    def __contains__(self, id: str) -> bool:
        return id in self.ids

    def __getitem__(self, idx: Union[int, str]) -> dict:
        if isinstance(idx, str):
            idx = self._id_to_idx[idx]

        data_dict = {k: v for k, v in zip(self.df.columns, self.df.values[idx])}

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: dict, transform: Callable[[Any], Any] = None):
        self._data_dict = data_dict
        self._idx_to_id = {idx: id for idx, id in enumerate(data_dict.keys())}
        self.transform = transform

    @cached_property
    def ids(self) -> Set[str]:
        return set(list(self._data_dict.keys()))

    def __len__(self) -> int:
        return len(self._data_dict)

    def __contains__(self, id: str) -> bool:
        return id in self._data_dict.keys()

    def __getitem__(self, idx: Union[int, str]) -> dict:
        if isinstance(idx, int):
            idx = self._idx_to_id[idx]

        data_dict = self._data_dict[idx]

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict

    @classmethod
    def preload_from_dataset(
        cls, dataset: torch.utils.data.Dataset, transform: Callable[[Any], Any] = None
    ):
        logger.info(f"Preloading dataset {dataset} into memory.")
        data_dict = {idx: dataset[idx] for idx in tqdm(range(len(dataset)))}
        return cls(data_dict, transform=transform)