import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import Parameter
from layers.HiPPOScale import HiPPOScale

class HHAVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim,device):
        super(HHAVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.device=device
        self.weights_pool = nn.Parameter(torch.ones(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.zeros(embed_dim, dim_out))
    def construct_adj(self,A, steps):
        """
        :param A: 空间邻接矩阵 (N, N)
        :param steps: 选择几个时间步来构建局部时空图
        :return: 局部时空图邻接矩阵 (steps x N, steps x N)
        """
        N = len(A)
        adj = np.zeros((N * steps, N * steps))
        for i in range(steps):
            # 对角线上是空间邻接矩阵
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A.cpu().detach().numpy()

        for i in range(N):
            for k in range(steps - 1):
                # 每个节点不同时间步与自身相连
                adj[k * N + i, (k + 1) * N + i] = 1
                adj[(k + 1) * N + i, k * N + i] = 1

        # for i in range(len(adj)):
        #     # 加入自环，在使用图卷积聚合信息时考虑自身特征
        #     adj[i, i] = 1
        return adj
    def forward(self, x, node_embeddings,adj):
        #x            shaped [B, N, C]
        #adj/supports shaped [N, N]
        #output       shaped [B, N, C]
        node_num = adj.shape[0]
        if x.shape[1]==node_num:
            supports = adj
            support_set = [torch.eye(node_num).to(supports.device), supports]
        elif x.shape[1]==3*node_num:
            supports = self.construct_adj(adj,3)
            supports=torch.tensor(supports).to(torch.float32).to(self.device)
            support_set = [torch.eye(3*node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        if x_g.shape[1]==3*node_num:
            x_g=x_g.reshape(x_g.shape[0],3,node_num,x_g.shape[2],x_g.shape[3])
            x_g=x_g[:,1,...]
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv

class HHAGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim,order,dropout,device):
        super(HHAGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.W_uh = nn.Linear(self.hidden_dim, 1)
        # self.meomory=HiPPOScale(order)
        self.order=order
        self.gate = HHAVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k,embed_dim,device)
        self.update = HHAVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k,embed_dim,device)
        

    def forward(self, x, state, c,t,node_embeddings,adj):
        #x:     B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        #c:     B,num_nodes,order

        state = state.to(x.device)
        c=c.to(x.device)
        # mx=torch.cat((c, x), dim=-1)
        # input_and_state = torch.cat((mx, state), dim=-1)
        if len(x.shape)==4:
            x=x.reshape(x.shape[0],3*self.node_num,x.shape[3])
        if x.shape[1]==self.node_num:
            # mx=torch.cat((c, x), dim=-1)
            input_and_state = torch.cat((x, state), dim=-1)
        elif x.shape[1]==3*self.node_num:
            # c=c.repeat(1, 3, 1)
            state=state.repeat(1,3,1)
            # mx=torch.cat((c, x), dim=-1)
            input_and_state = torch.cat((x, state), dim=-1)
        g = torch.sigmoid(self.gate(input_and_state,node_embeddings,adj))
        hc = torch.tanh(self.update(input_and_state,node_embeddings,adj))
        if x.shape[1]==3*self.node_num:
            # c=c[:,0::3,:]
            state=state[:,0::3,:]
        h = (1-g)*state + g*hc
        # f=self.W_uh(h)
        # c=self.meomory(c,f,t)
        c=c
        return h,c
         

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
    def init_hippo_c(self,batch_size):
        return torch.zeros(batch_size,self.node_num,self.order)