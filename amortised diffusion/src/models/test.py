import torch
from torch_geometric.data import Data, Batch


def get_mask(x, min_len, max_len):



def get_motif(x, masks):
    batch_sizes = x.batch.bincount().tolist()
    motif_positions = []
    
    mask_splits = torch.split(masks, batch_sizes)
    pos_splits = torch.split(x.pos, batch_sizes)
    
    for graph_mask, graph_pos in zip(mask_splits, pos_splits):
        motif_positions.append(graph_pos[graph_mask])
    
    return motif_positions


# Sample graphs for testing
graph1 = Data(pos=torch.randn(10, 3), edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]))
graph2 = Data(pos=torch.randn(15, 3), edge_index=torch.tensor([[6, 7, 8, 9], [7, 8, 9, 10]]))
graph3 = Data(pos=torch.randn(8, 3), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))

# Creating batch
x = Batch.from_data_list([graph1, graph2, graph3])

# Getting masks
masks = get_mask(x, 5, 8)

# Getting motif positions
motif_positions = get_motif(x, masks)

# Printing motif positions
for idx, pos in enumerate(motif_positions):
    print(f'Motif 3D positions in graph {idx}:\n {pos}')
