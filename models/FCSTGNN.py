import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from layers.FCSTGNNCell import Feature_extractor_1DCNN_RUL, GraphConvpoolMPNN_block_v6, PositionalEncoding, output_layer

class Model(nn.Module):
    def __init__(self, args, adj_mx, device):
        super(Model, self).__init__()
        if adj_mx is not None:
            raise Exception('STSGCN Model not need a pre-defined graph!')
        if args.features!='MS':
            raise Exception('FCSTGNN Model only concern multivariate predict univariate')
        self.horizon=args.pred_len
        self.nonlin_map = Feature_extractor_1DCNN_RUL(1, args.FCSTGNN_1DCNN_hidden_dim, args.FCSTGNN_1DCNN_output_dim,kernel_size=args.FCSTGNN_conv_kernel)
        self.nonlin_map_conv_out=args.input_dim-args.FCSTGNN_conv_kernel+4
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(args.FCSTGNN_1DCNN_output_dim*self.nonlin_map_conv_out, 2*args.FCSTGNN_hidden_dim),
            nn.BatchNorm1d(2*args.FCSTGNN_hidden_dim)
        )

        self.positional_encoding = PositionalEncoding(2*args.FCSTGNN_hidden_dim,0.1,max_len=5000)

        self.MPNN1 = GraphConvpoolMPNN_block_v6(2*args.FCSTGNN_hidden_dim, args.FCSTGNN_hidden_dim, args.num_node, 
        time_window_size=args.FCSTGNN_moving_window, stride=1, decay = args.FCSTGNN_decay, pool_choice=args.FCSTGNN_pooling_choice)


        self.predictLayer = nn.ModuleList()
        for t in range(self.horizon):
            self.predictLayer.append(
                output_layer(
                    num_of_vertices=args.num_node,
                    history=args.seq_len+1-args.FCSTGNN_moving_window,
                    in_dim=args.FCSTGNN_hidden_dim,
                    hidden_dim=args.FCSTGNN_out_layer_dim,
                    horizon=1
                )
            )

    def forward(self, X, x_mark_enc, x_dec, x_mark_dec, batches_seen=None):
        bs, tlen, num_node, dimension = X.size() 

        ### Graph Generation
        A_input = tr.reshape(X, [bs*tlen*num_node, dimension, 1])
        A_input_ = self.nonlin_map(A_input)
        A_input_ = tr.reshape(A_input_, [bs*tlen*num_node,-1])
        A_input_ = self.nonlin_map2(A_input_)
        A_input_ = tr.reshape(A_input_, [bs, tlen,num_node,-1])

        ## positional encoding before mapping starting
        X_ = tr.reshape(A_input_, [bs,tlen,num_node, -1])
        X_ = tr.transpose(X_,1,2)
        X_ = tr.reshape(X_,[bs*num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = tr.reshape(X_,[bs,num_node, tlen, -1])
        X_ = tr.transpose(X_,1,2)
        A_input_ = X_

        features = self.MPNN1(A_input_) 
        # return bs, num_windows,num_sensors, self.output_dim

        need_concat = []
        for i in range(self.horizon):
            out_step = self.predictLayer[i](features)  # (B, 1, N)
            need_concat.append(out_step)

        out = tr.cat(need_concat, dim=1)  # B, Tout, N

        out=tr.unsqueeze(out, dim=-1)    #B,Tout,N,1
        
        del need_concat

        return out