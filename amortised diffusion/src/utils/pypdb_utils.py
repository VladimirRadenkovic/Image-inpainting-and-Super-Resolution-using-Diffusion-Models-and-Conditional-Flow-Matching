import pypdb
import torch

def get_coordinates(protein_id, chain=None, device="cuda"):
    protein_data = pypdb.get_pdb_file(protein_id, filetype='pdb', compression=False)

    lines = protein_data.split('\n')
    atom_lines = [line for line in lines if line.startswith('ATOM')]

    N_coords = []
    CA_coords = []
    C_coords = []

    for line in atom_lines:
        atom_type = line[12:16].strip()
        chain_id = line[21].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])

        if chain is None or chain == chain_id:
            if atom_type == 'N':
                N_coords.append([x, y, z])
            elif atom_type == 'CA':
                CA_coords.append([x, y, z])
            elif atom_type == 'C':
                C_coords.append([x, y, z])

    N_coord_tensor = torch.tensor(N_coords, device=device)
    CA_coord_tensor = torch.tensor(CA_coords, device=device)
    C_coord_tensor = torch.tensor(C_coords, device=device)

    return N_coord_tensor, CA_coord_tensor, C_coord_tensor

def get_motif_coordinates(protein_id, motif_start, motif_end, chain=None):
    """
    Returns the coordinates of the CA atoms of the motif residues"""
    _, CA_coord, _ = get_coordinates(protein_id, chain)
    #center CA_coord
    CA_coord = CA_coord - CA_coord.mean(axis=0)
    #scale them down by factor 15
    CA_coord = CA_coord/15
    #get positions of motif range (numbering starts at 1 for PDB files)
    CA_coord = CA_coord[motif_start-1:motif_end]
    assert CA_coord.shape[0] == motif_end-motif_start+1, f"Motif range ({motif_end-motif_start+1}) does not match number of coordinates ({CA_coord.shape[0]}), check PDB file/chain ID"
    return CA_coord