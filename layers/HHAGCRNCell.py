import numpy as np
from scipy.sparse import coo_matrix
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import Parameter
from layers.HiPPOScale import HiPPOScale
from layers.HyperbolicCell import HGCNConv
from layers.manifolds.poincare import PoincareBall

class HHAVWGCN(nn.Module):
    def __init__(self, node_num,dim_in, dim_out, PoicareBall,cheb_k,embed_dim,dropout,c0,c1):
        super(HHAVWGCN, self).__init__()
        self.PoicareBall=PoicareBall
        self.c0=c0
        self.layer1 = HGCNConv(node_num,PoicareBall, dim_in, dim_out, cheb_k,embed_dim,c0, c1,
                                   dropout=dropout)
    def forward(self, x, node_embeddings,adj):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        #进行双曲图卷积
        self.c0=self.c0.to(x.device)
        x_tan = self.PoicareBall.proj_tan0(x, self.c0)
        x_hyp = self.PoicareBall.expmap0(x_tan, self.c0)
        x_hyp = self.PoicareBall.proj(x_hyp, self.c0)
        
        x_gconv=self.layer1.forward(x_hyp,node_embeddings,adj)
        return x_gconv

class HHAGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim,order,dropout,device):
        super(HHAGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.PoicareBall=PoincareBall()
        self.W_uh = nn.Linear(self.hidden_dim, 1)
        self.meomory=HiPPOScale(order)
        self.order=order
        self.hypebolic_c = Parameter(torch.ones(2, 1) * 1.0, requires_grad=True).to(device)
        self.gate = HHAVWGCN(node_num,dim_in+self.hidden_dim+self.order, dim_out, self.PoicareBall, cheb_k,embed_dim,dropout,self.hypebolic_c[0],self.hypebolic_c[1])
        self.update = HHAVWGCN(node_num,dim_in+self.hidden_dim+self.order, dim_out, self.PoicareBall, cheb_k,embed_dim,dropout,self.hypebolic_c[0],self.hypebolic_c[1])
        

    def forward(self, x, state, c, node_embeddings,t,adj):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        #c:B,num_nodes,order

        # turn to the hyperbolic 
        # x=self.toHyperX(x,self.hypebolic_c[0])
        # c=self.toHyperX(c, self.hypebolic_c[0])
        # state=self.toHyperX(state, self.hypebolic_c[0])

        self.hypebolic_c=self.hypebolic_c.to(x.device)
        state = state.to(x.device)
        c=c.to(x.device)
        mx=torch.cat((c, x), dim=-1)
        input_and_state = torch.cat((mx, state), dim=-1)
        g = torch.sigmoid(self.toTangentX(self.gate(input_and_state, node_embeddings,adj),self.hypebolic_c[0]))
        hc = torch.tanh(self.toTangentX(self.update(input_and_state, node_embeddings,adj),self.hypebolic_c[0]))
        
        # #turn back to the eduic
        # state=self.toTangentX(state,self.hypebolic_c[0])
        # c=self.toTangentX(c,self.hypebolic_c[0])

        h = (1-g)*state + g*hc
        f=self.W_uh(h)
        c=self.meomory(c,f,t)
        return h,c

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
    def init_hippo_c(self,batch_size):
        return torch.zeros(batch_size,self.node_num,self.order)
    
    def initHyperX(self, x, c=1.0):
        return self.toHyperX(x, c)

    def toHyperX(self, x, c=1.0):
        x_tan = self.PoicareBall.proj_tan0(x, c)
        x_hyp = self.PoicareBall.expmap0(x_tan, c)
        x_hyp = self.PoicareBall.proj(x_hyp, c)
        return x_hyp

    def toTangentX(self, x, c=1.0):
        x = self.PoicareBall.proj_tan0(self.PoicareBall.logmap0(x, c), c)
        return x