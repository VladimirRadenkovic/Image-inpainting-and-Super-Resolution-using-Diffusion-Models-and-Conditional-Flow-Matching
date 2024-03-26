import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
import random
import os
import numpy as np
import torch_geometric.utils
from sklearn.model_selection import train_test_split
import numpy as np


def get_cath_data(root_folder="/home/ked48/rds/hpc-work/protein-diffusion/data/cath_calpha"):

    pdb_train = PDBDatasetTrain(root=root_folder)
    pdb_val = PDBDatasetVal(root=root_folder)

    return pdb_train, pdb_val

def get_scope_data(root_folder="/home/ked48/rds/hpc-work/protein-diffusion/data/scope_calpha"):
    # list_files = [f for f in os.listdir(f"{root_folder}/raw") if f.endswith('.npy')]
    # print(f"Number of files: {len(list_files)}")
    # list_train, list_val = train_test_split(list_files, test_size=0.1, random_state=42)
    # print(f"Number of train files: {len(list_train)}")
    # print(f"Number of val files: {len(list_val)}")

    # scope_train = ScopeDatasetTrain(root=root_folder, list_files=list_train)
    # scope_val = ScopeDatasetVal(root=root_folder, list_files=list_val)
    scope_train = ScopeDatasetTrain(root=root_folder)
    scope_val = ScopeDatasetVal(root=root_folder)

    return scope_train, scope_val


class ScopeDataset(InMemoryDataset):
    def __init__(self, root, list_files=None, transform=None, pre_transform=None):
        if list_files:
            self.list_files = list_files
        else: #find all .npy files in the root folder
            self.list_files = [f for f in os.listdir(root) if f.endswith('.npy')]
        super(ScopeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.list_files

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self, max_size=256):
        data_list = []
        for prot_file in self.list_files:
            raw_path = os.path.join(self.raw_dir, prot_file)
            protein_pos = np.loadtxt(raw_path, delimiter=',')
            protein_pos = protein_pos - protein_pos.mean(axis=0)
            protein_pos = torch.from_numpy(protein_pos).to(torch.float)
            scaling_factor = 15.0  # Set your own scaling factor
            protein_pos = protein_pos / scaling_factor

            num_atoms = protein_pos.shape[0]
            edge_index = torch.ones(num_atoms, num_atoms)  # fully connected graph
            edge_index = edge_index.fill_diagonal_(0)  # remove self loops on diagonal
            edge_index = torch_geometric.utils.dense_to_sparse(edge_index)[0]
            node_order = torch.arange(num_atoms, dtype=torch.long) # node ordering induced by pos tensor
            protein_id = os.path.basename(raw_path).split('.')[0]  # Assuming file name is protein_id.npy
            new_graph = Data(
                pos=protein_pos, 
                edge_index=edge_index,
                node_order=node_order,
                protein_id = protein_id
            )
            if max_size:
                if num_atoms <= max_size:
                    data_list.append(new_graph)
            else:
                data_list.append(new_graph)

        data_list = random.sample(data_list, len(data_list))
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

class ScopeDatasetTrain(ScopeDataset):
    def processed_file_names(self):
        return ['scope_processed_ca_256max_train.pt']

class ScopeDatasetVal(ScopeDataset):
    def processed_file_names(self):
        return ['scope_processed_ca_256max_val.pt']



class PDBDatasetTrain(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['raw_train.npz']

    @property
    def processed_file_names(self):
        return ['processed_train.pt']

    def download(self):
        ...

    def process(self):

        # Read data into huge `Data` list.
        pdb = np.load(self.raw_paths[0])
        keys = list(pdb.keys())
        data_list = []

        scaling_factor = float(15)

        for k in keys:

            protein_pos = pdb[k]
            
            protein_pos = protein_pos - protein_pos.mean(axis=0)
            protein_pos = torch.from_numpy(protein_pos).to(torch.float)
            protein_pos = protein_pos / scaling_factor

            num_atoms = protein_pos.shape[0]
            edge_index = torch.ones(num_atoms, num_atoms)  # fully connected graph
            edge_index = edge_index.fill_diagonal_(0)  # remove self loops on diagonal
            edge_index = torch_geometric.utils.dense_to_sparse(edge_index)[0]
            node_order = torch.arange(num_atoms, dtype=torch.long) # node ordering induced by pos tensor
            new_graph = Data(
                pos=protein_pos, 
                edge_index=edge_index,
                node_order=node_order,
                protein_id = k
            )
            data_list.append(new_graph)

        data_list = random.sample(data_list, len(data_list))
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])    

class PDBDatasetVal(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['raw_val.npz']

    @property
    def processed_file_names(self):
        return ['processed_val.pt']

    def download(self):
        ...

    def process(self):

        # Read data into huge `Data` list.
        pdb = np.load(self.raw_paths[0])
        keys = list(pdb.keys())
        data_list = []

        scaling_factor = float(15)

        for k in keys:

            protein_pos = pdb[k]
            
            protein_pos = protein_pos - protein_pos.mean(axis=0)
            protein_pos = torch.from_numpy(protein_pos).to(torch.float)
            protein_pos = protein_pos / scaling_factor

            num_atoms = protein_pos.shape[0]
            edge_index = torch.ones(num_atoms, num_atoms)  # fully connected graph
            edge_index = edge_index.fill_diagonal_(0)  # remove self loops on diagonal
            edge_index = torch_geometric.utils.dense_to_sparse(edge_index)[0]
            node_order = torch.arange(num_atoms, dtype=torch.long) # node ordering induced by pos tensor
            new_graph = Data(
                pos=protein_pos, 
                edge_index=edge_index,
                node_order=node_order,
                protein_id = k
            )
            data_list.append(new_graph)

        data_list = random.sample(data_list, len(data_list))
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])    

def get_hinge(hinges_folder, hinge_ind):
    # for now only a single target!!!!
    hinge_ind = 0 # delete this later

    # hinge_folder = os.path.join(hinges_folder, f"hinge_{hinge_ind}")
    hinge_dict = np.load(os.path.join(hinge_folder, os.listdir(hinge_folder)[0]))

    # WARNING! CURRENT SETTING WORK ONLY FOR THE SINNGLE LYZ HINGE
    hinge_disp = torch.tensor(hinge_dict['topk_evec'],device='cuda')
    hinge_inds = torch.tensor(hinge_dict['topk_res'],device='cuda') - 20
    hinge_pos = torch.tensor(hinge_dict['topk_pos'],device='cuda') / 15

    return hinge_disp, hinge_inds, hinge_pos

if __name__ =='__main__':
    scope_train, scope_val = get_scope_data()
    pdb_train, pdb_val = get_cath_data()
    print(f"SCOPE Train: {len(scope_train)}")
    print(f"SCOPE Val: {len(scope_val)}")
    print(f"CATH Train: {len(pdb_train)}")
    print(f"CATH Val: {len(pdb_val)}")