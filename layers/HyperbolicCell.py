"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter, scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot, zeros
from layers.manifolds import PoincareBall

class HGCNConv(nn.Module):
    """
    Hyperbolic graph convolution layer, from hgcn。
    """

    def __init__(self, manifold, in_features, out_features, cheb_k,embed_dim,c_in=1.0, c_out=1.0, dropout=0.6, act=F.leaky_relu,
                 use_bias=True):
        super(HGCNConv, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout=dropout)
        self.agg = HypAgg(manifold, c_in, in_features,out_features, cheb_k,embed_dim,bias=use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold
        self.c_in = c_in

    def forward(self, x, node_embeddings):
        #h = self.linear.forward(x)
        h = self.agg.forward(x, node_embeddings)
        #h = self.hyp_act.forward(h)
        return h


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout=0.6, use_bias=True):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HypAgg(MessagePassing):
    """
    Hyperbolic aggregation layer using degree.
    """

    def __init__(self, manifold, c, in_features,out_features, cheb_k,embed_dim,bias=True):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.cheb_k=cheb_k
        self.embed_dim=embed_dim
        self.c = c
        self.weights_pool = nn.Parameter(torch.ones(embed_dim, cheb_k, in_features, out_features))
        self.bias_pool = nn.Parameter(torch.zeros(embed_dim, out_features))
    def forward(self, x, node_embeddings):
        x = self.manifold.logmap0(x, c=self.c)
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
        output = self.manifold.proj(self.manifold.expmap0(x_gconv, c=self.c), c=self.c)
        return output

