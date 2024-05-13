import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F

from layers.HHAGCRNwithoutHiPPOCell import HHAGCRNCell

class AVWDCRNN(nn.Module):
    def __init__(self,device, node_num, dim_in, dim_out, cheb_k, embed_dim, order,dropout,num_layers=1):
        super(AVWDCRNN, self).__init__()
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(HHAGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim,order,dropout,device))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(HHAGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim,order,dropout,device))

    def forward(self, x, init_state, init_hippo_c,node_embeddings,adj):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        #shape of init_hippo_c: (num_layers, B, N, order)
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            hippo_c=init_hippo_c[i]
            inner_states = []
            for t in range(seq_length):
                #state,hippo_c = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, hippo_c,t,node_embeddings,adj)
                if t==0 or t==seq_length-1:
                    state,hippo_c = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, hippo_c,t,node_embeddings,adj)
                else:
                    state,hippo_c = self.dcrnn_cells[i](current_inputs[:, t-1:t+2, :, :], state, hippo_c,t,node_embeddings,adj)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        init_hippo_c=[]
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
            init_hippo_c.append(self.dcrnn_cells[i].init_hippo_c(batch_size))
        return torch.stack(init_states, dim=0),torch.stack(init_hippo_c,dim=0)      
        #(num_layers, B, N, hidden_dim/order)

class Model(nn.Module):
    def __init__(self, args, adj_mx, device):
        # assert adj_mx==None
        assert args.output_dim==1
        super(Model, self).__init__()
        self.temperature=0.0001
        self.device=device
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.HHAGCRN_hidden_dim
        self.output_dim = args.output_dim
        self.horizon = args.pred_len 
        self.num_layers = args.HHAGCRN_num_layers
        self.embeded_dim = args.HHAGCRN_embed_dim
        
        self.encoder = AVWDCRNN(self.device,args.num_nodes, args.input_dim, args.HHAGCRN_hidden_dim, args.HHAGCRN_cheb_k,
                                args.HHAGCRN_embed_dim, args.HHorder,args.dropout,args.HHAGCRN_num_layers)

        self.end_conv = nn.Conv2d(1, args.pred_len * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        
        
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.HHAGCRN_embed_dim), requires_grad=True)
        self.node_embeddings2 = nn.Parameter(torch.randn(self.num_node, args.HHAGCRN_embed_dim), requires_grad=True)
        self.fc_out = nn.Linear(self.embeded_dim * 2, self.embeded_dim)
        self.fc_cat = nn.Linear(self.embeded_dim, 2)
        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot
        # Generate off-diagonal interaction graph
        off_diag = np.ones([self.num_node, self.num_node])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(self.device)
        self.rel_send = torch.FloatTensor(rel_send).to(self.device)
    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)
    
    def sample_gumbel(self,shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self,logits, temperature, eps=1e-10):
        sample = self.sample_gumbel(logits.size(), eps=eps)
        sample=sample.to(logits.device)
        y = logits + sample
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self,logits, temperature, hard=False, eps=1e-10):
        y_soft = self.gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
        if hard:
            shape = logits.size()
            _, k = y_soft.data.max(-1)
            y_hard = torch.zeros(*shape).to(k.device)
            y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
            y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
        else:
            y = y_soft
        return y
    def forecast(self, source,node_embeddings,adj):
        # shape (batch_size, seq_len, num_sensor, input_dim)
        # Normalization from Non-stationary Transformer
        means = source.mean(1, keepdim=True).detach()
        source = source - means
        stdev = torch.sqrt(torch.var(source, dim=1, keepdim=True, unbiased=False) + 1e-5)
        source /= stdev        
        #source: B, T_1, N, D
        #target: B, T_2, N, D

        init_state,init_hippo_c = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, init_hippo_c,node_embeddings,adj)      #B, T, N, hidden
        
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, 1, self.num_node)
        output = output.permute(0, 1, 3, 2)    #B,T,N,1
        #采样
        mu=0
        logvar=0
        #B, T, N, C
        output = output * (stdev[:, 0, :, -1:].unsqueeze(1).repeat(1, self.horizon, 1,1))
        output = output + (means[:, 0, :, -1:].unsqueeze(1).repeat(1, self.horizon, 1,1))

        return output,mu,logvar


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batches_seen=None):
        # 时间季节特征
        # x_enc = torch.cat([x_enc, x_mark_enc.unsqueeze(2).expand(-1, -1, self.num_node, -1)], dim=-1)
        receivers = torch.matmul(self.rel_rec, self.node_embeddings)
        senders = torch.matmul(self.rel_send, self.node_embeddings)
        x = torch.cat([senders, receivers], dim=1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)
        # (num_nodes*num_nodes,2)

        adj = self.gumbel_softmax(x, temperature=self.temperature, hard=True)
        adj = adj[:, 0].clone().reshape(self.num_node, -1)
        mask = torch.eye(self.num_node, self.num_node).bool().to(self.device)
        adj.masked_fill_(mask, 0)
        dec_out = self.forecast(x_enc,self.node_embeddings,adj)
        return dec_out