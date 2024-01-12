import numpy as np
import torch
import torch.nn as nn

from layers.DCRNNCell import DCGRUCell


class Seq2SeqAttrs:
    def __init__(self, args, adj_mx, device):
        self.adj_mx = adj_mx
        self.max_diffusion_step = args.cheb_k
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = args.use_curriculum_learning
        self.num_nodes = args.num_nodes
        self.num_layers = args.num_layers
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.hidden_dim = args.hidden_dim
        self.hidden_state_size = self.num_nodes * self.hidden_dim
        self.filter_type = args.filter_type
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.batch_size = args.batch_size
        self.device = device


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args, adj_mx, device):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args, adj_mx, device)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.hidden_dim, self.adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_layers)])

    def forward(self, inputs, hidden_state=None):
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
            hidden_state = torch.zeros((self.num_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args, adj_mx, device):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args, adj_mx, device)
        self.projection_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.hidden_dim, self.adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_layers)])

    def forward(self, inputs, hidden_state):
        """
        Decoder forward pass.

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
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.hidden_dim))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class Model(nn.Module, Seq2SeqAttrs):
    def __init__(self, args, adj_mx, device):
        if adj_mx is None:
            raise Exception('DCRNN Model need a pre-defined graph!')
        super().__init__()
        Seq2SeqAttrs.__init__(self, args, adj_mx, device)
        self.encoder_model = EncoderModel(args, adj_mx, device)
        self.decoder_model = DecoderModel(args, adj_mx, device)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.pred_len):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
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
        """
        seq2seq forward pass
        :param batch_x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param batch_x_mark
        :param dec_inp: shape (batch_size, horizon, num_sensor, input_dim)
        :param batch_y_mark
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        x, y = self.prepare_data(batch_x, dec_inp)
        encoder_hidden_state = self.encoder(x)

        outputs = self.decoder(encoder_hidden_state, y, batches_seen=batches_seen)
        # output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        outputs = outputs.permute(1, 0, 2)
        outputs = torch.reshape(outputs, (self.batch_size, self.pred_len, self.num_nodes, self.output_dim))
        return outputs

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
