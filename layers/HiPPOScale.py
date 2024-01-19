import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy import linalg as la
import math
class HiPPOScale(nn.Module):
    """ Vanilla HiPPO-LegS model (scale invariant instead of time invariant) """
    def __init__(self, N, max_length=1024):
        """
        max_length: maximum sequence length
        """
        super().__init__()
        self.N = N
        A, B = self.transition(N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
            B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
        self.register_buffer('A_stacked', torch.Tensor(A_stacked)) # (max_length, N, N)
        self.register_buffer('B_stacked', torch.Tensor(B_stacked)) # (max_length, N)
        print(self.B_stacked[2].shape)

    def transition(selff,N):
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
        return A,B
    
    def forward(self,c,f,t):
        #c:B,num_nodes,order
        #f:B, num_nodes, 1
        c = F.linear(c, self.A_stacked[t]) + self.B_stacked[t] * f
        return c