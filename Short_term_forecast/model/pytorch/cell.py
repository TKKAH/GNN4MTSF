import numpy as np
import torch
from lib import utils
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape,name='param'):
        if shape not in self._params_dict:
            if name=='node_embeddings':
                nn_param = torch.nn.Parameter(torch.zeros(*shape, device=device))
            else:
                nn_param = torch.nn.Parameter(torch.zeros(*shape, device=device))
                # torch.nn.init.xavier_normal_()是一个用于对张量进行 Xavier 正态分布初始化的函数。
                #torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            # 调用torch.nn.init.constant_(biases, bias_start)会将biases张量中的所有元素设置为bias_start的值。
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, max_diffusion_step, num_nodes, nonlinearity='tanh',
                use_gc_for_ru=False):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')
        self.node_embeddings = nn.Parameter(torch.randn(self._num_nodes, 10), requires_grad=True)
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
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A.cpu().detach().numpy()

        for i in range(N):
            for k in range(steps - 1):
                # 每个节点不同时间步与自身相连
                adj[k * N + i, (k + 1) * N + i] = 1
                adj[(k + 1) * N + i, k * N + i] = 1

        for i in range(len(adj)):
            # 加入自环，在使用图卷积聚合信息时考虑自身特征
            adj[i, i] = 1
        return adj
    def _calculate_random_walk_matrix(self, adj_mx):

        # tf.Print(adj_mx, [adj_mx], message="This is adj: ")

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        # 这个操作的目的是处理逆矩阵中可能存在的无穷值情况。
        # 在计算逆矩阵时，如果某个元素的倒数为无穷大（inf），则将其替换为零。
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        # DI**（-1）A**T
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, inputs, hx, adj):
        """Gated recurrent unit (GRU) with Graph Convolution
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        hidden state
        """
        adj_mx = self._calculate_random_walk_matrix(adj).t()
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, adj_mx, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        # c = self._gconv(inputs, adj_mx, r * hx, self._num_units)
        c=fn(inputs, adj_mx, r * hx, self._num_units)
        c = torch.reshape(c, (-1, self._num_nodes, self._num_units))
        c=c.reshape(-1, self._num_nodes*self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, adj_mx,state, output_size, bias_start=0.0):
        if len(inputs.shape)==2:
            batch_size = inputs.shape[0]
            inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
            state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        elif len(inputs.shape)==3:
            batch_size = inputs.shape[1]
            inputs = torch.reshape(inputs, (3*batch_size * self._num_nodes, -1))
            state=state.unsqueeze(0)
            state=state.repeat(3,1,1)
            state = torch.reshape(state, (3*batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.matmul(inputs_and_state, weights)
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        if value.shape[0]==3*batch_size*self._num_nodes:
            value=value.reshape(3,batch_size*self._num_nodes,-1)
            value=value[1]
        return value

    def _gconv(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        if len(inputs.shape)==2:
            batch_size = inputs.shape[0]
            inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
            state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        elif len(inputs.shape)==3:
            batch_size = inputs.shape[1]
            inputs = torch.reshape(inputs, (3,batch_size, self._num_nodes, -1))
            state=state.unsqueeze(0)
            state=state.repeat(3,1,1)
            state = torch.reshape(state, (3,batch_size,self._num_nodes, -1))

        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.size(-1)
        x = inputs_and_state
        if len(x.shape)==3:
            x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)

            x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        elif len(x.shape)==4:
            x0 = x.permute(0,2, 3, 1)  # (num_nodes, total_arg_size, batch_size)

            x0 = torch.reshape(x0, shape=[3*self._num_nodes, input_size * batch_size])
            adj_mx=self.construct_adj(adj_mx,3)
            adj_mx=torch.tensor(adj_mx).to(torch.float32).to(device)
        x = torch.unsqueeze(x0, 0)
        
        if self._max_diffusion_step == 0:
            pass
        else:
            x1 = torch.mm(adj_mx, x0)
            x = self._concat(x, x1)

            for k in range(2, self._max_diffusion_step + 1):
                x2 = 2 * torch.mm(adj_mx, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1
        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.
        if x.shape[1]==3*self._num_nodes:
            x=x.reshape(x.shape[0],3,self._num_nodes,x.shape[2])
            x=x[:,1,...]
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 0, 2)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
        # weights_pool = self._gconv_params.get_weights((10,num_matrices,input_size , output_size),name='node_embeddings')
        # biases_pool = self._gconv_params.get_weights((10,output_size),name='node_embeddings')  
        # weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, weights_pool)  #N, cheb_k, dim_in, dim_out
        # bias = torch.matmul(self.node_embeddings, biases_pool)                       #N, dim_out
        # x_gconv = torch.einsum('bnki,nkio->bno', x, weights) + bias     #b, N, dim_out
        # return torch.reshape(x_gconv, [batch_size, self._num_nodes * output_size])