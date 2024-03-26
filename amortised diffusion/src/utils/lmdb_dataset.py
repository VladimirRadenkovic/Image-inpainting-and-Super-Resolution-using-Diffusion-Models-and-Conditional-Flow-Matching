from __future__ import annotations

import gzip
import io
import pathlib
import pickle as pkl
from typing import Any, Callable, Iterable, Literal, Sequence, Tuple

import lmdb
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

try:
    import msgpack
except ImportError:
    logger.warning("msgpack not installed, cannot use msgpack serialisation")
try:
    import dill
except ImportError:
    logger.warning("dill not installed, cannot use dill serialisation")
try:
    import biotite.structure.io.npz as npz
except ImportError:
    logger.warning("biotite not installed, cannot serealise biotite serialisation")
try:
    import torch
except ImportError:
    logger.warning("torch not installed, cannot serealise torch objects")

__all__ = ["LMDBDataset"]


LMDB_MAP_SIZE = 10_000_000_000_000  # 10 TB


class LMDBDataset:
    """
    Creates a dataset from an lmdb file.
    """

    def __init__(self, lmdb_path: pathlib.Path | str, transform=None):
        lmdb_path = pathlib.Path(lmdb_path).absolute()
        if not lmdb_path.exists():
            raise FileNotFoundError(lmdb_path)
        self._lmdb_path = lmdb_path

        self._connect()
        with self._env.begin(write=False) as txn:
            self._serialisation_format = _lmdb_get("serialisation_format", txn)
            self._compresslevel = int(_lmdb_get("compresslevel", txn))
            self._metadata = self._cast_to_type(
                _lmdb_get("metadata", txn, self._serialisation_format),
                str(pd.DataFrame),
            )
        # NOTE: We remove the `_env` variable in `init` on purpose as it messes with
        #   multiprocessing (cannot be pickled). We re-connect in `__getitem__` as needed.
        #   c.f. https://github.com/pytorch/vision/issues/689#issuecomment-787215916
        self._disconnect()
        self._transform = transform
        self._size_in_bytes = None

    # ==================== Basic Properties ====================
    def __len__(self) -> int:
        return len(self._metadata)

    def __contains__(self, id: str) -> bool:
        return id in self._metadata.index

    @property
    def lmdb_path(self) -> pathlib.Path:
        return self._lmdb_path

    @property
    def name(self) -> str:
        return self.lmdb_path.stem

    @property
    def metadata(self) -> None | pd.DataFrame:
        return self._metadata

    @property
    def ids(self) -> set:
        return set(self._metadata.index)

    @property
    def index(self) -> pd.Index[str]:
        return self._metadata.index

    @property
    def serialisation_format(self) -> str:
        return self._serialisation_format

    def size_on_disk(self, unit: Literal["B", "KB", "MB", "GB", "TB"] = "MB") -> float:
        units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}
        self._size_in_bytes = (self.lmdb_path / "data.mdb").stat().st_size
        return self._size_in_bytes / units[unit]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.lmdb_path})"
            f"\n\tname:          {self.name}"
            f"\n\tnum_examples:  {len(self):,}"
            f"\n\tsize_on_disk:  {self.size_on_disk(unit='MB'):,.2f} MB"
        )

    # ==================== Getter functions ====================
    def _connect(
        self,
        readonly: bool = True,
        max_readers: int = 1,
        lock: bool = False,
        readahead: bool = False,
        meminit: bool = False,
        **kwargs,
    ) -> None:
        self._env = lmdb.open(
            str(self.lmdb_path),
            max_readers=max_readers,
            readonly=readonly,
            lock=lock,
            readahead=readahead,
            meminit=meminit,
            **kwargs,
        )

    def _disconnect(self) -> None:
        if self._env:
            self._env = None

    def _which_idx(self, id: str) -> int:
        return self._metadata.index.get_loc(id)

    def id_to_idx(self, ids: str | list[str]) -> list[int]:
        if isinstance(ids, str):
            return self._which_idx(ids)
        return [self._which_idx(id) for id in ids]

    def idx_to_id(self, idx: int | list[int]) -> list[str]:
        if isinstance(idx, int):
            return self.metadata.index.values[idx]
        return self.metadata.index.values[idx]

    def __getitem__(self, index_or_id: int | str) -> dict:
        """Get item by index or id."""
        if isinstance(index_or_id, int):
            index_or_id = self.idx_to_id(index_or_id)

        if self._env is None:
            self._connect()

        with self._env.begin(write=False) as txn:
            try:
                item = _lmdb_get(str(index_or_id), txn, self._serialisation_format)
            except KeyError as e:
                raise e
            except Exception as e:
                logger.warning("Error getting item %s: %s" % (str(index_or_id), str(e)))
                return None

        # Recover special data types
        if "_types" in item.keys():
            for x in item.keys():
                item[x] = self._cast_to_type(item[x], item["_types"][x])
            item.pop("_types")  # remove internal _types key
        else:
            logger.warning(
                "Data types in item %i not defined. Will use basic types only." % index_or_id
            )

        if self._transform:
            item = self._transform(item)
        return item

    def update_metadata(self, metadata: pd.DataFrame, **env_kwargs) -> "LMDBDataset":
        """
        Update metadata. Useful if you want to add new columns to the metadata or
        wanted to change the order of columns or rows. All ids must be the same.

        Args:
            metadata (pd.DataFrame): New metadata.

        Returns:
            LMDBDataset: self
        """
        assert set(self.ids) == set(metadata.index), "ids must be the same"
        env = lmdb.open(str(self.lmdb_path), map_size=LMDB_MAP_SIZE, lock=True, **env_kwargs)
        with env.begin(write=True) as txn:
            _lmdb_set(
                key="metadata",
                val=metadata,
                txn=txn,
                serialisation_format=self.serialisation_format,
                compresslevel=self._compresslevel,
                overwrite=True,
            )
            logger.info("Successfully updated metadata.")
        self._metadata = metadata
        return self

    def delete_data(self, ids: str | Sequence[str]) -> "LMDBDataset":
        """
        Delete data from the dataset.
        WARNING: This will permanently delete the data from the dataset. It will not
        free up the space on disk due to LMDB's memory mapping. To free up the space,
        you will need to copy the database with `compact = True`.
        c.f.
            - https://blogs.kolabnow.com/2018/06/07/a-short-guide-to-lmdb
            - https://lmdb.readthedocs.io/en/release/#lmdb.Environment.copy

        Args:
            ids (str | Sequence[str]): Ids to delete.

        Returns:
            LMDBDataset: self
        """
        if isinstance(ids, str):
            ids = [ids]
        # Ensure we only try to delete ids that actually exist
        ids_to_delete = set(ids).intersection(self.ids)

        deleted_ids = set()
        env = lmdb.open(str(self.lmdb_path), map_size=LMDB_MAP_SIZE, lock=True)
        with env.begin(write=True) as txn:
            for id in tqdm(ids_to_delete, total=len(ids_to_delete)):
                try:
                    _lmdb_del(id, txn)
                    deleted_ids.add(id)
                except Exception as e:
                    logger.warning("Error deleting item %s: %s" % (id, str(e)))
            new_metadata = self.metadata.drop(index=list(deleted_ids))
            _lmdb_set(
                key="metadata",
                val=new_metadata,
                txn=txn,
                serialisation_format=self.serialisation_format,
                compresslevel=self._compresslevel,
                overwrite=True,
            )
            self._metadata = new_metadata

        logger.info(f"Successfully deleted {len(deleted_ids)}/{len(ids)} entries.")
        return self

    def add_data(
        self,
        dataset: dict | Sequence[dict],
        metadata: pd.DataFrame,
        overwrite: bool = False,
        compresslevel: int = 6,
        **env_kwargs,
    ) -> "LMDBDataset":
        """
        Add data with corresponding metadata to the dataset.

        Args:
            dataset (dict | Sequence[dict]): data to add.
            metadata (pd.DataFrame): metadata to add.
            overwrite (bool): whether to overwrite existing data. Default: False.

        Returns:
            LMDBDataset: self
        """
        assert isinstance(metadata, pd.DataFrame)
        if isinstance(dataset, dict):
            dataset = [dataset]

        env = lmdb.open(str(self.lmdb_path), map_size=LMDB_MAP_SIZE, **env_kwargs)

        added_ids = set()
        metadata_ids = set(metadata.index)
        existing_ids = set(self.metadata.index)
        with env.begin(write=True) as txn:
            # Add all datapoints to database
            for x in tqdm(dataset, total=len(dataset)):
                try:
                    assert overwrite or x["id"] not in existing_ids, "id already exists in database"
                    assert x["id"] in metadata_ids, "id must be in metadata index"
                    assert x["id"] not in added_ids, "id must be unique"
                    # Add an entry that stores the original types of all entries
                    x["_types"] = {key: str(type(val)) for key, val in x.items()}
                    # ... including itself
                    x["_types"]["_types"] = str(type(x["_types"]))
                    # Add to database
                    _lmdb_set(
                        key=x["id"],
                        val=x,
                        txn=txn,
                        serialisation_format=self.serialisation_format,
                        compresslevel=compresslevel,
                        overwrite=overwrite,
                    )
                    added_ids.add(x["id"])
                except Exception as e:
                    logger.warning("Error adding item %s: %s" % (str(x["id"]), str(e)))

            if len(added_ids) > 0:
                new_metadata = pd.concat([self.metadata, metadata.loc[list(added_ids)]])
                _lmdb_set(
                    key="metadata",
                    val=new_metadata,
                    txn=txn,
                    serialisation_format=self.serialisation_format,
                    compresslevel=self._compresslevel,
                    overwrite=True,
                )
                self._metadata = new_metadata
        logger.info(f"Successfully added {len(added_ids)}/{len(dataset)} values.")
        return self

    def detect_integrity_problems(self) -> dict:
        if self._env is None:
            self._connect()
        with self._env.begin() as txn:
            keys = set([key.decode() for key in txn.cursor().iternext(values=False)])

        for key in {"metadata", "serialisation_format", "compresslevel"}:
            keys.remove(key)  # Raises KeyError if key is not in keys

        integrity_problems = {
            "missing_metadata": keys - set(self.metadata.index),
            "missing_data": set(self.metadata.index) - keys,
        }

        n_missing_metadata = len(integrity_problems["missing_metadata"])
        n_missing_data = len(integrity_problems["missing_data"])

        if n_missing_metadata == 0 and n_missing_data == 0:
            logger.info("No integrity problems detected.")
            return {}
        if n_missing_metadata > 0:
            logger.warning(f"Found {n_missing_metadata} missing metadata entries.")
        if n_missing_data > 0:
            logger.warning(f"Found {n_missing_data} missing data entries.")
        return integrity_problems

    # ==================== Creation functions ====================
    @staticmethod
    def create(
        lmdb_path: pathlib.Path | str,
        dataset: Sequence[dict],
        metadata: pd.DataFrame,
        serialisation_format: Literal["dill", "pkl", "msgpack"] = "msgpack",
        map_size: int = LMDB_MAP_SIZE,  # 10 TB
        compresslevel: int = 6,
    ) -> "LMDBDataset":
        """
        Create a new LMDBDataset from a dataset and metadata and save it to disk.

        Args:
            lmdb_path (pathlib.Path | str): path under which to create the lmdb dataset.
            dataset (Sequence[dict]): data to add. Each dict must have an "id" key. Ids must be unique.
            metadata (pd.DataFrame): metadata to add. Ids of the metadata must be unique and
                match the ids of the dataset.
            serialisation_format (Literal["dill", "pkl", "msgpack"]): serialisation format. Default: "msgpack".
            map_size (int): size of the lmdb map. Default: 10 TB.

        Returns:
            LMDBDataset: self
        """
        lmdb_path = pathlib.Path(lmdb_path).absolute()
        if lmdb_path.exists():
            raise FileExistsError(f"lmdb_path {lmdb_path} exists already")

        logger.info(f"Creating lmdb dataset with {len(dataset)} examples at `{lmdb_path}`")
        # Check that all ids are unique
        assert metadata.index.is_unique, "ids must be unique"
        # Create lmdb environment (creates database)
        env = lmdb.open(str(lmdb_path), map_size=map_size)

        # Fill database
        with env.begin(write=True) as txn:
            added_ids = set()
            metadata_ids = set(metadata.index)
            _lmdb_set(
                key="serialisation_format", val=serialisation_format, txn=txn, compresslevel=0
            )
            _lmdb_set(key="compresslevel", val=compresslevel, txn=txn, compresslevel=0)
            # Add all datapoints to database
            for x in tqdm(dataset, total=len(dataset)):
                try:
                    assert x["id"] in metadata_ids, "id must be in metadata index"
                    assert x["id"] not in added_ids, "id must be unique"
                    # Add an entry that stores the original types of all entries
                    x["_types"] = {key: str(type(val)) for key, val in x.items()}
                    # ... including itself
                    x["_types"]["_types"] = str(type(x["_types"]))
                    # Add to database
                    _lmdb_set(
                        key=x["id"],
                        val=x,
                        txn=txn,
                        serialisation_format=serialisation_format,
                        compresslevel=compresslevel,
                    )
                    added_ids.add(x["id"])
                except Exception as e:
                    logger.warning("Error adding item %s: %s" % (str(x["id"]), str(e)))
            # Update metadata to only contain added ids
            metadata = metadata.loc[list(added_ids)]
            _lmdb_set(
                key="metadata",
                val=metadata,
                txn=txn,
                serialisation_format=serialisation_format,
                compresslevel=min(6, compresslevel),
            )
        logger.info(
            f"Dataset creation completed. {len(added_ids)}/{len(dataset)} successfully added."
        )
        return LMDBDataset(lmdb_path)

    @staticmethod
    def build_from_stream(
        lmdb_path: pathlib.Path | str,
        samples: Iterable["sample"],
        get_id: Callable[["sample"], str],
        get_data: Callable[["sample"], Tuple[dict, dict]],
        buffer_size: int = 100,
    ) -> "LMDBDataset":
        """
        Build a LMDBDataset from a stream of samples. This is useful for building a dataset
        from a large number of samples that cannot be stored in memory. The samples are
        processed one by one and added to the buffer. The buffer is saved to disk after
        every `buffer_size` samples.

        Args:
            lmdb_path (pathlib.Path | str): path under which to create the lmdb dataset. Can be
                an existing dataset, in which case the new samples are added to the existing
                dataset.
            samples (Iterable["sample"]): stream of samples to add to the dataset.
            get_id (Callable[["sample"], str]): function that returns the id of a sample.
            get_data (Callable[["sample"], Tuple[dict, dict]]): function that returns a tuple
                of (datapoint, metadata) for a sample. Both, datapoint and metadata, must be
                dicts with an "id" key that matches the id returned by `get_id` for the sample.
            buffer_size (int): number of samples to add to the buffer before saving it to disk.
                Default: 100.

        Returns:
            LMDBDataset: The dataset with the new samples added.
        """
        try:
            db = LMDBDataset(lmdb_path)
        except:
            db = LMDBDataset.create(
                lmdb_path=lmdb_path,
                dataset=[],
                metadata=pd.DataFrame(),
                serialisation_format="msgpack",
            )

        dataset, metadata = [], []
        for sample in tqdm(samples):
            # Save buffer
            if len(dataset) >= buffer_size:
                metadata = pd.DataFrame(metadata)
                metadata.set_index("id", inplace=True)
                db.add_data(dataset=dataset, metadata=metadata, lock=True)
                dataset = []
                metadata = []

            try:
                id = get_id(sample)
                if id in db:
                    continue
                # Add datapoint to buffer
                datapoint, meta = get_data(sample)
                dataset.append(datapoint)
                metadata.append(meta)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.warning(f"{id} failed: {e}")

        # Save last buffer
        if len(dataset) > 0:
            metadata = pd.DataFrame(metadata)
            metadata.set_index("id", inplace=True)
            db.add_data(dataset=dataset, metadata=metadata, lock=True)

        return db

    def _cast_to_type(self, obj: Any, to_type: str) -> Any:
        """Helper function to avoid casting already casted objects from pickle serialisation"""
        if self._serialisation_format in ("pkl", "dill"):
            return obj
        else:
            return _cast_to_type(obj, to_type=to_type)


# =================== LMDB Helper functions ===================
def _lmdb_set(
    key: str | bytes,
    val: object,
    txn: lmdb.Transaction,
    serialisation_format: Literal["pkl", "dill", "msgpack"] | None = None,
    compresslevel: int = 0,
    overwrite: bool = False,
) -> bool:
    """Helper function to write objects to lmdb database"""
    key: bytes = key if isinstance(key, bytes) else str(key).encode()
    # Encode
    val: bytes = serialise(val, serialisation_format)
    # Compress
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=compresslevel) as f:
        f.write(val)
    val = buf.getvalue()
    # Save
    result: bool = txn.put(key, val, overwrite=overwrite)
    # Validate that writing was successful
    if not result:
        raise RuntimeError(f"LMDB entry {key} already exists")
    return result


def _lmdb_get(
    key: str | bytes,
    txn: lmdb.Transaction,
    serialisation_format: Literal["pkl", "dill", "msgpack"] | None = None,
) -> object:
    """Helper function to retrieve items from a given lmdb database"""
    key: bytes = key if isinstance(key, bytes) else key.encode()
    # Retrieve
    val: bytes = txn.get(key)
    if val is None:
        raise KeyError(f"Key {key.decode()} not found in database")
    # Decompress
    buf = io.BytesIO(val)
    with gzip.GzipFile(fileobj=buf, mode="rb") as f:
        val = f.read()
    # Decode
    val: object = deserialise(val, serialisation_format)
    return val


def _lmdb_del(key: str | bytes, txn: lmdb.Transaction) -> bool:
    """Helper function to retrieve items from a given lmdb database"""
    key: bytes = key if isinstance(key, bytes) else key.encode()
    return txn.delete(key)


# =================== Serialisation functions ===================
def serialise(x: object, serialisation_format: Literal["pkl", "dill", "msgpack"]) -> bytes:
    """
    Serialises dataset `x` in format given by `serialisation_format` (pkl, dill, msgpack).
    """
    if serialisation_format == "pkl":
        # pickle: Brittle across languages/python versions.
        return pkl.dumps(x)
    elif serialisation_format == "dill":
        # dill: Memory efficient and more robust than pickle.
        return dill.dumps(x)
    elif serialisation_format == "msgpack":
        # msgpack: A bit more memory efficient than json, a bit less supported.
        serialised = msgpack.packb(x, default=_prepare_serialisation)
    elif serialisation_format == None:
        serialised = x if isinstance(x, bytes) else str(x).encode()
    else:
        raise RuntimeError("Invalid serialization format %s" % serialisation_format)
    return serialised


def deserialise(x: bytes, serialisation_format: Literal["pkl", "dill", "msgpack"] | None) -> object:
    """
    Deserialises dataset `x` assuming format given by `serialisation_format` (pkl, dill, msgpack).
    """
    if serialisation_format == "pkl":
        return pkl.loads(x)
    elif serialisation_format == "dill":
        return dill.loads(x)
    elif serialisation_format == "msgpack":
        serialised = msgpack.unpackb(x, strict_map_key=False)
    elif serialisation_format == None:
        serialised = x.decode() if isinstance(x, bytes) else x
    else:
        raise RuntimeError("Invalid serialisation format %s" % serialisation_format)
    return serialised


def _prepare_serialisation(obj: object) -> object:
    """Custom serealisation methods for objects that are not serialisable by default."""
    if isinstance(obj, np.ndarray):
        with io.BytesIO() as buf:
            np.save(buf, obj)
            return buf.getvalue()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="split")
    elif isinstance(obj, torch.Tensor):  # TODO: Switch to string comparison?
        return _prepare_serialisation(obj.detach().cpu().numpy())
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif (
        str(type(obj)) == "<class 'biotite.structure.AtomArray'>"
    ):  # NOTE: We perform string comparison to avoid requiring biotite as a dependency
        with io.BytesIO() as buf:
            f = npz.NpzFile()
            f.set_structure(obj)
            f.write(buf)
            return buf.getvalue()
    else:
        raise RuntimeError("Cannot serialise object of type %s" % type(obj))


def _cast_to_type(obj: Any, to_type: str) -> Any:
    """Helper function to cast a (deserialised) object to a custom python type"""
    if to_type == str(pd.DataFrame):
        return pd.DataFrame(**obj)
    elif to_type == str(np.ndarray):
        try:
            return np.load(io.BytesIO(obj))
        except:
            return np.array(obj)
    elif to_type == "<class 'torch.Tensor'>":
        return torch.Tensor(_cast_to_type(obj, str(np.ndarray)))
    elif (
        to_type == "<class 'biotite.structure.AtomArray'>"
    ):  # Use str here to avoid requiring biotite import.
        try:
            return npz.NpzFile.read(io.BytesIO(obj)).get_structure()
        except:
            return obj
    return obj