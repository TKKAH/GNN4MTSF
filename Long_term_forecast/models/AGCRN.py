import torch
import torch.nn as nn
from layers.AGCRNCell import AGCRNCell

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class Model(nn.Module):
    def __init__(self, args, adj_mx, device):
        assert adj_mx==None
        assert args.output_dim==1
        super(Model, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.AGCRN_hidden_dim
        self.output_dim = args.output_dim
        self.horizon = args.pred_len 
        self.num_layers = args.AGCRN_num_layers
        self.embeded_dim = args.AGCRN_embed_dim

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.AGCRN_embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.AGCRN_hidden_dim, args.AGCRN_cheb_k,
                                args.AGCRN_embed_dim, args.AGCRN_num_layers)

        self.end_conv = nn.Conv2d(1, args.pred_len * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forecast(self, source):
        # shape (batch_size, seq_len, num_sensor, input_dim)
        # Normalization from Non-stationary Transformer
        means = source.mean(1, keepdim=True).detach()
        source = source - means
        stdev = torch.sqrt(torch.var(source, dim=1, keepdim=True, unbiased=False) + 1e-5)
        source /= stdev

        #source: B, T_1, N, D
        #target: B, T_2, N, D

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)
        #B, T, N, C
        output = output * (stdev[:, 0, :, -1:].unsqueeze(1).repeat(1, self.horizon, 1,1))
        output = output + (means[:, 0, :,-1:].unsqueeze(1).repeat(1, self.horizon, 1,1))
        return output


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batches_seen=None):
        dec_out = self.forecast(x_enc)
        return dec_out