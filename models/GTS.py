import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from layers.GTSCell import DCGRUCell

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
  if hard:
      shape = logits.size()
      _, k = y_soft.data.max(-1)
      y_hard = torch.zeros(*shape).to(device)
      y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
      y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
  else:
      y = y_soft
  return y

class Seq2SeqAttrs:
    def __init__(self, args, adj_mx, device):
        self.max_diffusion_step = args.GTS_cheb_k
        self.cl_decay_steps = args.GTS_cl_decay_steps
        self.use_curriculum_learning = args.GTS_use_curriculum_learning
        self.num_nodes = args.num_nodes
        self.num_rnn_layers = args.GTS_num_layers
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.hidden_dim = args.GTS_hidden_dim
        self.hidden_state_size = self.num_nodes * self.hidden_dim
        self.filter_type = args.GTS_filter_type
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.batch_size = args.batch_size
        self.device = device
        self.temperature = args.GTS_temperature
        self.embedding_dim = 100

class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args, adj_mx, device):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args, adj_mx, device)
        
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.hidden_dim, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args, adj_mx, device):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args, adj_mx, device)
        self.projection_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.hidden_dim, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):
        """
        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.hidden_dim))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class Model(nn.Module, Seq2SeqAttrs):
    def __init__(self, args, adj_mx, device):
        assert adj_mx is not None
        assert args.output_dim==1
        super().__init__()
        Seq2SeqAttrs.__init__(self, args, adj_mx[0], device)
        self.node_fea=adj_mx[1]
        self.encoder_model = EncoderModel(args, adj_mx[0], device)
        self.decoder_model = DecoderModel(args, adj_mx[0], device)
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)  # .to(device)
        self.hidden_drop = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(16*(self.node_fea.shape[1]-9-9), self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot
        # Generate off-diagonal interaction graph
        off_diag = np.ones([self.num_nodes, self.num_nodes])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(self.device)
        self.rel_send = torch.FloatTensor(rel_send).to(self.device)


    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], adj, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.pred_len):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, batch_x, batch_x_mark, dec_inp=None, batch_y_mark=None, batches_seen=None):
               # inputs, node_feas, labels=None, batches_seen=None):
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        x = self.node_fea.view(self.num_nodes, 1, -1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = x.view(self.num_nodes, -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.bn3(x)

        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)
        # (num_nodes*num_nodes,2)

        adj = gumbel_softmax(x, temperature=self.temperature, hard=True)
        adj = adj[:, 0].clone().reshape(self.num_nodes, -1)
        mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(device)
        adj.masked_fill_(mask, 0)

        inputs, labels = self.prepare_data(batch_x, dec_inp)
        encoder_hidden_state = self.encoder(inputs, adj)
        outputs = self.decoder(encoder_hidden_state, adj, labels, batches_seen=batches_seen)
         # output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        outputs = outputs.permute(1, 0, 2)
        outputs = torch.reshape(outputs, (self.batch_size, self.pred_len, self.num_nodes, self.output_dim))
        return outputs, x.softmax(-1)[:, 0].clone().reshape(self.num_nodes, -1)
    
    def prepare_data(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        x = x.permute(1, 0, 2, 3)
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        if y is not None:
            y = y.permute(1, 0, 2, 3)
            y = y[..., :self.output_dim].view(self.pred_len, batch_size,
                                              self.num_nodes * self.output_dim)
            return x.to(self.device), y.to(self.device)
        else:
            return x.to(self.device), None