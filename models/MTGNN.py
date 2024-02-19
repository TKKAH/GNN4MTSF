from layers.MTGNN_layer import *


class Model(nn.Module):
    def __init__(self, args, adj_mx, device):
        assert adj_mx is None
        assert args.output_dim==1 
        super(Model, self).__init__()
        self.gcn_true = True
        self.buildA_true = True
        self.num_nodes = args.num_nodes
        self.dropout = args.dropout
        self.pred_len=args.pred_len
        self.predefined_A = None
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=args.input_dim,
                                    out_channels=args.MTGNN_residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(args.num_nodes, args.MTGNN_top_k_graph, args.MTGNN_node_embedding_dim, 'cuda:0')

        self.seq_length = args.seq_len
        kernel_size = 7
        if args.MTGNN_dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(args.MTGNN_dilation_exponential**args.MTGNN_layers-1)/(args.MTGNN_dilation_exponential-1))
        else:
            self.receptive_field = args.MTGNN_layers*(kernel_size-1) + 1

        for i in range(1):
            if args.MTGNN_dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(args.MTGNN_dilation_exponential**args.MTGNN_layers-1)/(args.MTGNN_dilation_exponential-1))
            else:
                rf_size_i = i*args.MTGNN_layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,args.MTGNN_layers+1):
                if args.MTGNN_dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(args.MTGNN_dilation_exponential**j-1)/(args.MTGNN_dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(args.MTGNN_residual_channels, args.MTGNN_conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(args.MTGNN_residual_channels, args.MTGNN_conv_channels, dilation_factor=new_dilation))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=args.MTGNN_conv_channels,
                                                    out_channels=args.MTGNN_skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=args.MTGNN_conv_channels,
                                                    out_channels=args.MTGNN_skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(args.MTGNN_conv_channels, args.MTGNN_residual_channels, args.MTGNN_gcn_depth, args.dropout, args.MTGNN_propalpha))
                    self.gconv2.append(mixprop(args.MTGNN_conv_channels, args.MTGNN_residual_channels, args.MTGNN_gcn_depth, args.dropout, args.MTGNN_propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((args.MTGNN_residual_channels,args.num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=True))
                else:
                    self.norm.append(LayerNorm((args.MTGNN_residual_channels, args.num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=True))

                new_dilation *= args.MTGNN_dilation_exponential

        self.layers = args.MTGNN_layers
        self.end_conv_1 = nn.Conv2d(in_channels=args.MTGNN_skip_channels,
                                             out_channels=args.MTGNN_end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=args.MTGNN_end_channels,
                                             out_channels=args.pred_len,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=args.input_dim, out_channels=args.MTGNN_skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=args.MTGNN_residual_channels, out_channels=args.MTGNN_skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=args.input_dim, out_channels=args.MTGNN_skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=args.MTGNN_residual_channels, out_channels=args.MTGNN_skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to('cuda:0')


    def forecast(self, input,idx=None):
        # BTNC
        means = input.mean(1, keepdim=True).detach()
        input = input - means
        stdev = torch.sqrt(torch.var(input, dim=1, keepdim=True, unbiased=False) + 1e-5)
        input /= stdev


        
        input=input.transpose(3,1)
        ##B,C,N,T

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
        #adp:(N,N)
        x = self.start_conv(input)#(B,C,N,T)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training)) #(B,C,N,1)
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
            skip = s + skip     #(B,C,N,1)

            x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip #(B,C,N,1)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)


        # B, T, N, C
        output = output * (stdev[:, 0, :, -1:].unsqueeze(1).repeat(1, self.horizon, 1,1))
        output = output + (means[:, 0, :, -1:].unsqueeze(1).repeat(1, self.horizon, 1,1))
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batches_seen=None):
        dec_out = self.forecast(x_enc)
        return dec_out#[:, -self.pred_len:, :，F]  # [B, L, N,D]