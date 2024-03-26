import gzip
import io
import os
import pathlib
from typing import Dict, Union

import biotite.structure as bs
import biotite.structure.io.mmtf as mmtf
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
from Bio.Data.IUPACData import protein_letters_3to1

_3to1 = _3to1 = {key.upper(): val for key, val in protein_letters_3to1.items()}


def structure_from_buffer(buffer: io.StringIO, file_type="cif", **load_kwargs) -> bs.AtomArray:
    buffer.seek(0)
    if file_type in ("cif", "mmcif", "pdbx"):
        file = pdbx.PDBxFile()
        file.read(buffer)
        if "assembly_id" in load_kwargs:
            return pdbx.get_assembly(file, **load_kwargs)
        return pdbx.get_structure(file, **load_kwargs)
    elif file_type in ("pdb", "pdb1"):
        file = pdb.PDBFile()
        file.read(buffer)
        return pdb.get_structure(file, **load_kwargs)
    elif file_type == "mmtf":
        file = mmtf.MMTFFile()
        file.read(buffer)
        return mmtf.get_structure(file, **load_kwargs)
    else:
        raise ValueError(f"Unknown type: {file_type}")


def load_structure(
    path_or_buffer: Union[io.StringIO, pathlib.Path, str], file_type: str = None, **load_kwargs
) -> bs.AtomArray:
    if isinstance(path_or_buffer, io.StringIO):
        assert file_type is not None, "Type must be specified when loading from buffer"
        return structure_from_buffer(path_or_buffer, file_type=file_type, **load_kwargs)

    path = pathlib.Path(path_or_buffer)
    assert path.exists(), f"File does not exist: {path}"

    if path.suffix in (".gz", ".gzip"):
        open_func = gzip.open
        file_type = os.path.splitext(path.stem)[-1].split(".")[-1]
    else:
        open_func = open
        file_type = path.suffix.split(".")[-1]

    buffer = io.BytesIO() if file_type == "mmtf" else io.StringIO()
    mode = "rb" if file_type == "mmtf" else "rt"
    with open_func(path, mode) as file:
        buffer.write(file.read())
    return structure_from_buffer(buffer, file_type=file_type, **load_kwargs)


def get_sequence(structure: bs.AtomArray) -> Dict[str, str]:
    structure = structure[bs.filter_amino_acids(structure)]
    sequence = {}
    for chain_id in bs.get_chains(structure):
        sequence_array = bs.residues.get_residues(structure[structure.chain_id == chain_id])[1]
        sequence[chain_id] = "".join([_3to1[res] for res in sequence_array])
    return sequence


def atom_array_from_numpy(coords: np.ndarray) -> bs.AtomArray:
    """Convert a numpy array of coordinates to a biotite AtomArray, inserting ALA residues as dummies.
    Useful for visualising pure coordinate backbones in PyMOL.
    """
    n_atoms = coords.shape[0]
    atom_array = bs.AtomArray(n_atoms)
    atom_array.coord = coords
    atom_array.chain_id = ["A"] * n_atoms
    atom_array.res_name = ["ALA"] * n_atoms
    atom_array.atom_name = ["CA"] * n_atoms
    atom_array.element = ["C"] * n_atoms
    atom_array.res_id = np.arange(1, n_atoms + 1)
    return atom_array

def ca_backbone_from_atom_array(atom_array: bs.AtomArray) -> np.ndarray:
    """Extract the coordinates of the CA atoms from an AtomArray.
    """
    return atom_array.coord[atom_array.atom_name == "CA"]