import numpy as np
# import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.MSGNetCell import Attention_Block, GraphBlock, Predict


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class ScaleGraphBlock(nn.Module):
    def __init__(self, args, adj_mx, device):
        super(ScaleGraphBlock, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.k = args.MSGNet_FTT_top_k

        self.att0 = Attention_Block(args.MSGNet_enc_dim, 
                                    n_heads=args.MSGNet_n_heads, dropout=args.dropout, activation="gelu")
        self.norm = nn.LayerNorm(args.MSGNet_enc_dim)
        self.gelu = nn.GELU()
        self.gconv = nn.ModuleList()
        for i in range(self.k):
            self.gconv.append(
                GraphBlock(args.num_nodes , args.MSGNet_enc_dim ,  args.MSGNet_conv_channel,  args.MSGNet_skip_channel,
                        args.MSGNet_gcn_depth , args.dropout,  args.MSGNet_propalpha ,args.seq_len,
                           args.MSGNet_node_dim))


    def forward(self, x):
        B, T, N = x.size()
        scale_list, scale_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            scale = scale_list[i]
            #Gconv
            x = self.gconv[i](x)
            # x (B,T,dmodel)
            # paddng
            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = out.reshape(B, length // scale, scale, N)

        #for Mul-attetion
            out = out.reshape(-1 , scale , N)
            out = self.norm(self.att0(out))
            out = self.gelu(out)
            out = out.reshape(B, -1 , scale , N).reshape(B ,-1 ,N)
            out = out[:, :self.seq_len, :]
            res.append(out)

        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    def __init__(self, args, adj_mx, device):
        super(Model, self).__init__()
        assert adj_mx is None
        assert args.output_dim==1
        assert args.input_dim==1
        
        self.device = device
        self.model = nn.ModuleList([ScaleGraphBlock(args, adj_mx, device) for _ in range(args.MSGNet_enc_layers)])
        self.enc_embedding = DataEmbedding(args.num_nodes, args.MSGNet_enc_dim,
                                           args.embed, args.freq, args.dropout)
        self.layer = args.MSGNet_enc_layers
        self.layer_norm = nn.LayerNorm(args.MSGNet_enc_dim)

        self.predict_linear = nn.Linear(
            args.seq_len, args.pred_len + args.seq_len)
        self.projection = nn.Linear(
            args.MSGNet_enc_dim, args.num_nodes, bias=True)
        self.seq2pred = Predict(args.MSGNet_Predict_Layer_individual,args.num_nodes,
                                args.seq_len, args.pred_len, args.dropout)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batches_seen=None):
        # x_enc (b,T,N,dim)
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.squeeze(dim=-1)
        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  
        # enc_out [B,T,dmodel]
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # porject back
        dec_out = self.projection(enc_out)
        # dec_out [B,T,N]
        dec_out = self.seq2pred(dec_out.transpose(1, 2)).transpose(1, 2)
        
        dec_out = dec_out.unsqueeze(dim=-1)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :,:].unsqueeze(1).repeat(
                      1, self.pred_len, 1,1))
        dec_out = dec_out + \
                  (means[:, 0, :,:].unsqueeze(1).repeat(
                      1, self.pred_len, 1,1))

        return dec_out[:, -self.pred_len:, :,:]