import numpy as np
from scipy.sparse import coo_matrix
from layers.manifolds.poincare import PoincareBall
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter, scatter_add
def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot
off_diag = np.ones([10,10])
print(encode_onehot(np.where(off_diag)[0]))
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
