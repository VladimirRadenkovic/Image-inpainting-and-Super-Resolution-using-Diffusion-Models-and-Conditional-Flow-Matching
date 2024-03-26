import re

import biotite.structure as bs  # noqa
import torch
import torch_geometric
import numpy as np

from typing import Sequence

from src.utils.biotite_utils import AA_3_TO_INT, load_structure


class LoadMMTF:
    def __init__(self, mmtf_dataset, pdb_field="pdb", structure_field="structure", **load_kwargs):
        self.mmtf_dataset = mmtf_dataset
        self.pdb_field = pdb_field
        self.structure_field = structure_field
        self.load_kwargs = load_kwargs

    def __call__(self, data_dict: dict) -> dict:
        pdb = data_dict[self.pdb_field]
        data_dict[self.structure_field] = load_structure(
            self.mmtf_dataset[pdb], file_type="mmtf", **self.load_kwargs
        )
        return data_dict


class FilterByArrayChain:
    def __init__(self, chain_field="chain", structure_field="structure"):
        self.structure_field = structure_field
        self.chain_field = chain_field

    def __call__(self, data_dict: dict) -> dict:
        structure = data_dict[self.structure_field]
        chain = data_dict[self.chain_field]
        data_dict[self.structure_field] = structure[structure.chain_id == chain]
        return data_dict


class FilterByAtomName:
    def __init__(self, allowed_atom_names: Sequence[str], structure_field="structure"):
        self.structure_field = structure_field
        self.allowed_atom_names = np.asarray(allowed_atom_names)

    def __call__(self, data_dict: dict) -> dict:
        structure = data_dict[self.structure_field]
        data_dict[self.structure_field] = structure[
            np.isin(structure.atom_name, self.allowed_atom_names)
        ]
        return data_dict


class FilterByElement:
    def __init__(self, allowed_elements: Sequence[str], structure_field="structure"):
        self.structure_field = structure_field
        self.allowed_elements = np.asarray(allowed_elements)

    def __call__(self, data_dict: dict) -> dict:
        structure = data_dict[self.structure_field]
        data_dict[self.structure_field] = structure[
            np.isin(structure.element, self.allowed_elements)
        ]
        return data_dict


class FilterByFunction:
    def __init__(self, filter_function, structure_field="structure"):
        self.structure_field = structure_field
        self.filter_function = filter_function

    def __call__(self, data_dict: dict) -> dict:
        structure = data_dict[self.structure_field]
        data_dict[self.structure_field] = structure[self.filter_function(structure)]
        return data_dict


class FilterBackbone(FilterByFunction):
    def __init__(self, structure_field="structure"):
        super().__init__(bs.filter_peptide_backbone, structure_field)


class FilterCanonicalAminoAcids(FilterByFunction):
    def __init__(self, structure_field="structure"):
        super().__init__(bs.filter_canonical_amino_acids, structure_field)


class FilterAminoAcids(FilterByFunction):
    def __init__(self, structure_field="structure"):
        super().__init__(bs.filter_amino_acids, structure_field)


class DropFields:
    def __init__(self, fields_to_drop):
        self.fields_to_drop = fields_to_drop

    def __call__(self, data_dict: dict) -> dict:
        data_dict = {k: v for k, v in data_dict.items() if k not in self.fields_to_drop}
        return data_dict

class ConvertToGraph:
    # TODO: Enable atom-level graphs with a grouping feature by residue id
    def __init__(self, structure_field="structure", drop_rest: bool = True):
        self.structure_field = structure_field
        self.drop_rest = drop_rest

    def __call__(self, data_dict: dict) -> dict:
        structure = data_dict[self.structure_field]
        data_dict[self.structure_field] = torch_geometric.data.Data(
            pos=torch.from_numpy(structure.coord),
            res=torch.tensor([AA_3_TO_INT[res] for res in structure.res_name], dtype=torch.long),
            **{k: v for k, v in data_dict.items() if k != self.structure_field},
        )
        if self.drop_rest:
            return data_dict[self.structure_field]
        return data_dict
    
    


# TODO:
# class ConstructResidueFrames
# class Construct AF2Features
# class Construct DihedralFeatures