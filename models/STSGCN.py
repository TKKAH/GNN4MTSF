import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from layers.STSGCNCell import STSGCL, output_layer

class Model(nn.Module):
    def __init__(self, args, adj_mx, device):
        assert adj_mx is not None
        assert args.output_dim==1
        super(Model, self).__init__()
        self.adj = self.construct_adj(adj_mx,args.STSGCN_strides)
        self.num_of_vertices = args.num_nodes
        self.hidden_dims = args.STSGCN_hidden_dims
        self.out_layer_dim = args.STSGCN_out_layer_dim
        self.activation = args.STSGCN_activation
        self.use_mask = args.STSGCN_use_mask

        self.temporal_emb = True
        self.spatial_emb = True
        self.horizon = args.pred_len
        self.strides = args.STSGCN_strides

        self.First_FC = nn.Linear(args.input_dim, args.STSGCN_first_layer_embedding_size, bias=True)
        self.STSGCLS = nn.ModuleList()
        self.STSGCLS.append(
            STSGCL(
                adj=self.adj,
                history=args.seq_len,
                num_of_vertices=self.num_of_vertices,
                in_dim=args.STSGCN_first_layer_embedding_size,
                out_dims=self.hidden_dims[0],
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb
            )
        )

        in_dim = self.hidden_dims[0][-1]
        history -= (self.strides - 1)

        for idx, hidden_list in enumerate(self.hidden_dims):
            if idx == 0:
                continue
            self.STSGCLS.append(
                STSGCL(
                    adj=self.adj,
                    history=history,
                    num_of_vertices=self.num_of_vertices,
                    in_dim=in_dim,
                    out_dims=hidden_list,
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb
                )
            )

            history -= (self.strides - 1)
            in_dim = hidden_list[-1]

        self.predictLayer = nn.ModuleList()
        for t in range(self.horizon):
            self.predictLayer.append(
                output_layer(
                    num_of_vertices=self.num_of_vertices,
                    history=history,
                    in_dim=in_dim,
                    hidden_dim=args.STSGCN_out_layer_dim,
                    horizon=1
                )
            )

        if self.use_mask:
            mask = torch.zeros_like(self.adj)
            mask[self.adj != 0] = self.adj[self.adj != 0]
            self.mask = nn.Parameter(mask)
        else:
            self.mask = None

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
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

        for i in range(N):
            for k in range(steps - 1):
                # 每个节点不同时间步与自身相连
                adj[k * N + i, (k + 1) * N + i] = 1
                adj[(k + 1) * N + i, k * N + i] = 1

        for i in range(len(adj)):
            # 加入自环，在使用图卷积聚合信息时考虑自身特征
            adj[i, i] = 1
        return adj
    
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, batches_seen=None):
        """
        :param x: B, Tin, N, Cin)
        :return: B, Tout, N
        """

        x = torch.relu(self.First_FC(x))  # B, Tin, N, Cin

        for model in self.STSGCLS:
            x = model(x, self.mask)
        # (B, T - 8, N, Cout)

        need_concat = []
        for i in range(self.horizon):
            out_step = self.predictLayer[i](x)  # (B, 1, N)
            need_concat.append(out_step)

        out = torch.cat(need_concat, dim=1)  # B, Tout, N

        out=torch.unsqueeze(out, dim=-1)    #B,Tout,N,1
        
        del need_concat

        return out