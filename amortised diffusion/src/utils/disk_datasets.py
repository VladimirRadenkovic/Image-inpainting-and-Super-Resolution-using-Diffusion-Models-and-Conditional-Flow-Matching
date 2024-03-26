import pathlib
import tarfile
import zipfile
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, List, Literal, Union

import msgpack
import numpy as np
from loguru import logger
from torch.utils.data import Dataset

from src.utils.lmdb_dataset import LMDBDataset

__all__ = ["FileDataset", "TarDataset", "ZipDataset"]


class OnDiskDataset(ABC, Dataset):
    @abstractmethod
    def __init__(self):
        """Must set self._root and self._ids in __init__."""

    @abstractmethod
    def get_by_id(self, item: str) -> Any:
        pass

    def _complete_setup(self):
        assert hasattr(self, "_root"), "Must set self._root in __init__"
        assert hasattr(self, "_ids"), "Must set self._ids in __init__"

        self._root = pathlib.Path(self._root)
        self._ids = np.asarray(self._ids)

        # Check that root exists
        if not self._root.exists():
            raise FileNotFoundError(f"Root directory {self.root.absolute()} does not exist.")
        # Check that there are some ids
        if len(self._ids) == 0:
            raise RuntimeError(f"No ids found in root directory {self.root}.")
        super().__init__()

    def get_id(self, item: Union[str, int]) -> List[str]:
        if isinstance(item, str):
            return item
        else:
            return self._ids[item]

    @cached_property
    def ids(self) -> set:
        return set(self._ids)

    @property
    def root(self) -> pathlib.Path:
        return pathlib.Path(self._root)

    def __getitem__(self, idx_or_id: Union[str, int]) -> Any:
        key = self.get_id(idx_or_id)
        if key not in self:
            raise KeyError(f"Item {key} not in {self}.")
        val = self.get_by_id(key)
        if hasattr(self, "transform") and self.transform is not None:
            val = self.transform(val)
        return val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root={self.root.absolute()!r}, n={len(self):,})"

    def __contains__(self, item: Union[str, int]) -> bool:
        return item in self.ids

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> iter:
        # Create iterator returning (id, item) tuples
        return zip(self.ids, (self[i] for i in range(len(self))))


class FileDataset(OnDiskDataset):
    VALID_READ_MODES = ("r", "rb", "path")

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        suffix: str = None,
        read_mode: Literal["r", "rb", "path"] = "rb",
        ids: List[str] = None,
        transform: callable = None,
    ):
        self._root = pathlib.Path(root)
        self._read_mode = read_mode
        self._suffix = suffix
        self.transform = transform

        # Check existence of root and that it is a directory
        if not self._root.exists():
            raise FileNotFoundError(f"Root directory {self._root} does not exist.")
        if not self._root.is_dir():
            raise NotADirectoryError(f"Root path {self._root} is not a directory.")
        # Check read mode
        if self._read_mode not in self.VALID_READ_MODES:
            raise ValueError(f"Read mode {self._read_mode} is not valid.")

        # Set ids (extract only the filenames)
        self._ids = []
        for path in self._root.glob(f"*.{suffix}" if suffix is not None else "*"):
            # Split at first dot
            name, ext = path.stem, path.suffix[1:]  # path.name.split(".", 1)
            if ids is not None and name not in ids:
                continue

            # Check if suffix is already set
            if self._suffix is None:
                self._suffix = ext
            # Check if suffix is the same for all files
            elif self._suffix != ext:
                raise ValueError(f"Found different file extensions in {self.root}.")

            self._ids.append(name)

        # If ids are given, check that all ids are present
        if ids is not None:
            missing_ids = set(ids) - set(self._ids)
            if len(missing_ids) > 0:
                raise ValueError(f"Could not find all ids in {self.root}: {missing_ids}.")
            self._ids = ids

        self._complete_setup()

    def get_by_id(self, item: Union[str, int]) -> Union[pathlib.Path, bytes, str]:
        if self._read_mode == "path":
            return self.root / f"{item}.{self._suffix}"
        with open(file=self.root / f"{item}.{self._suffix}", mode=self._read_mode) as f:
            return f.read()


class TarDataset(OnDiskDataset):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        suffix: str = None,
        use_index: bool = True,
        transform: callable = None,
    ):
        self._root = pathlib.Path(root)
        self._suffix = suffix
        self.transform = transform

        # Check existence of root and that it is a tar file
        if not self.root.exists():
            raise FileNotFoundError(f"Root tar file {self.root} does not exist.")
        if not tarfile.is_tarfile(self.root):
            raise ValueError(f"Root path {self.root} is not a tar file.")

        # If index exists, load it
        if use_index and (self.root.with_suffix(".index").exists()):
            logger.info("Loading tar index from disk.")
            with open(self.root.with_suffix(".index"), "rb") as index_file:
                tar_index = msgpack.load(index_file)
        # Otherwise, create it
        else:
            tar_index = self.create_tar_index(tar_path=self.root)
            logger.info("Saving tar index to disk.")
            with open(self.root.with_suffix(".index"), "wb") as index_file:
                msgpack.dump(tar_index, index_file)

        # Extract ids and offsets from index
        self._ids = []
        self._offsets = {}
        for path, offset in tar_index.items():
            # Split at first dot
            name, ext = path.split(".", 1)
            if suffix is None or ext == f"{suffix}":
                self._ids.append(name)
                self._offsets[name] = offset

        self._complete_setup()

    @staticmethod
    def create_tar_index(tar_path: Union[str, pathlib.Path]):
        tar_index = {}
        tar_path = pathlib.Path(tar_path)
        logger.info(
            "Extracting member offsets from tar file. This may take a while. Please wait... (per 100k files: ~1min)"
        )
        with tarfile.open(tar_path, "r") as tar_file:
            for member in tar_file.getmembers():
                if member.isfile():
                    tar_index[member.name] = member.offset
        return tar_index

    def get_by_id(self, item: str) -> Union[pathlib.Path, bytes, str]:
        with open(self._root, "rb") as fileobj:
            fileobj.seek(self._offsets[item])  # Move the file pointer to the stored offset
            with tarfile.open(fileobj=fileobj, mode="r|") as t:
                member = t.next()  # Get the member at the current file pointer position
                with t.extractfile(member) as f:
                    return f.read()


class ZipDataset(OnDiskDataset):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        suffix: str = None,
        transform: callable = None,
    ):
        self._root = pathlib.Path(root)
        self._suffix = suffix
        self.transform = transform

        # Check existence of root and that it is a zip file
        if not self.root.exists():
            raise FileNotFoundError(f"Root zip file {self.root} does not exist.")
        if not zipfile.is_zipfile(self.root):
            raise ValueError(f"Root path {self.root} is not a zip file.")

        self._ids = []
        with zipfile.ZipFile(self.root, "r") as archive:
            # Set ids (extract only the filenames without extensions)
            for path in archive.infolist():
                # Split at first dot
                name, ext = path.filename.split(".", 1)
                if self._suffix is None:
                    self._suffix = ext
                elif self._suffix != ext:
                    raise ValueError(
                        f"Found different file extensions in {self.root}: {ext} vs {self._suffix}."
                    )
                self._ids.append(name)

        # Check that there are files in root
        if len(self._ids) == 0:
            raise FileNotFoundError(f"No relevant files found in root zip file {self.root}.")

        self._complete_setup()

    def get_by_id(self, item: str) -> Union[pathlib.Path, bytes, str]:
        with zipfile.ZipFile(self.root, "r") as z:
            with z.open(f"{item}.{self._suffix}", "r") as f:
                return f.read()


def get_dataset(root: Union[str, pathlib.Path], **kwargs) -> type:
    root = pathlib.Path(root)
    if root.is_dir() and not root.suffix:
        return FileDataset(root, **kwargs)
    elif root.suffix == ".lmdb":
        return LMDBDataset(root, **kwargs)
    elif tarfile.is_tarfile(root):
        return TarDataset(root, **kwargs)
    elif zipfile.is_zipfile(root, **kwargs):
        return ZipDataset(root, **kwargs)
    else:
        raise ValueError(f"Could not determine dataset type for {root}.")
