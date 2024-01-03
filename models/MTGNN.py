from layers.MTGNN_layer import *


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.gcn_true = config.gcn_true
        self.buildA_true = config.buildA_true
        self.num_nodes = config.num_nodes
        self.dropout = config.dropout_MTGNN
        self.pred_len=config.pred_len
        self.predefined_A = None
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=config.input_dim,
                                    out_channels=config.residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(config.num_nodes, config.subgraph_size, config.node_dim, 'cuda:0', alpha=config.tanhalpha, static_feat=None)

        self.seq_length = config.seq_len
        kernel_size = 7
        if config.dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(config.dilation_exponential**config.layers-1)/(config.dilation_exponential-1))
        else:
            self.receptive_field = config.layers*(kernel_size-1) + 1

        for i in range(1):
            if config.dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(config.dilation_exponential**config.layers-1)/(config.dilation_exponential-1))
            else:
                rf_size_i = i*config.layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,config.layers+1):
                if config.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(config.dilation_exponential**j-1)/(config.dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(config.residual_channels, config.conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(config.residual_channels, config.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=config.conv_channels,
                                                    out_channels=config.residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=config.conv_channels,
                                                    out_channels=config.skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=config.conv_channels,
                                                    out_channels=config.skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(config.conv_channels, config.residual_channels, config.gcn_depth, config.dropout_MTGNN, config.propalpha))
                    self.gconv2.append(mixprop(config.conv_channels, config.residual_channels, config.gcn_depth, config.dropout_MTGNN, config.propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((config.residual_channels,config. num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=True))
                else:
                    self.norm.append(LayerNorm((config.residual_channels, config.num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=True))

                new_dilation *= config.dilation_exponential

        self.layers = config.layers
        self.end_conv_1 = nn.Conv2d(in_channels=config.skip_channels,
                                             out_channels=config.end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=config.end_channels,
                                             out_channels=config.output_dim*config.pred_len,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=config.input_dim, out_channels=config.skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=config.residual_channels, out_channels=config.skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=config.input_dim, out_channels=config.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=config.residual_channels, out_channels=config.skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to('cuda:0')


    def forecast(self, input,idx=None):
        # BTN
        means = input.mean(1, keepdim=True).detach()
        input = input - means
        stdev = torch.sqrt(torch.var(input, dim=1, keepdim=True, unbiased=False) + 1e-5)
        input /= stdev
        # print('input shape:',input.shape)
        _, L, N = input.shape


        if(len(input.shape)==3):
            input=input.unsqueeze(-1)
            input=input.transpose(3,1)
        ##B,C,N,T
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))



        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        output=x.squeeze()


        # B, T, N, C
        output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return output


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        dec_out = self.forecast(x_enc)
        return dec_out#[:, -self.pred_len:, :]  # [B, L, D]

        # return None