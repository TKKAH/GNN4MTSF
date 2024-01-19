import torch
import torch.nn.functional as F
import torch.nn as nn

from layers import HiPPOScale

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.ones(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.zeros(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)

        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv

class HippoAGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim,order):
        super(HippoAGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.W_uh = nn.Linear(self.hidden_dim, 1)
        self.meomory=HiPPOScale(order)
        self.order=order
        self.gate = AVWGCN(dim_in+self.hidden_dim+self.order, dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim+self.order, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, c, node_embeddings,t):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        #c:B,num_nodes,order
        state = state.to(x.device)
        c=c.to(x.device)
        mx=torch.cat((c, x), dim=-1)
        input_and_state = torch.cat((mx, state), dim=-1)
        g = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        hc = torch.tanh(self.update(input_and_state, node_embeddings))
        h = (1-g)*state + g*hc
        f=self.W_uh(h)
        c=self.meomory(c,f,t)
        return h,c

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
    def init_hippo_c(self,batch_size):
        return torch.zeros(batch_size,self.node_num,self.order)