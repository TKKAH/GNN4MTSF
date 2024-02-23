import numpy as np
from scipy.sparse import coo_matrix
from layers.manifolds.poincare import PoincareBall
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter, scatter_add
a=PoincareBall()
x=torch.randn((32,10,2))
batch_size=x.shape[0]
x=x.reshape((x.shape[1],-1))
def norm(edge_index, num_nodes, edge_weight):
    fill_value = 1 
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

supports=torch.randn((10,10))
supports = coo_matrix(supports)
indices = np.vstack((supports.row, supports.col))  
edge_index = torch.LongTensor(indices)  
edge_weight=torch.FloatTensor(supports.data)
edge_index, norm = norm(edge_index, x.size(0), edge_weight)
node_i = edge_index[0]
node_j = edge_index[1]
x_j = torch.nn.functional.embedding(node_j, x)
support = norm.view(-1, 1) * x_j
support_t = scatter(support, node_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
support_t=support_t.reshape((batch_size,support_t.shape[0],-1))
print(support_t.shape)
