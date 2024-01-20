import torch
import torch.nn as nn

from layers.MTGATModule import ConvLayer, FeatureAttentionLayer, Forecasting_Model, GRULayer, TemporalAttentionLayer



class Model(nn.Module):
    def __init__(
        self,
        args, adj_mx, device
    ):
        super(Model, self).__init__()

        self.conv = ConvLayer(args.num_nodes, args.kernel_size)
        self.feature_gat = FeatureAttentionLayer(args.num_nodes, args.seq_len, args.dropout, args.MTGAT_alpha, args.feat_gat_embed_dim, args.use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(args.num_nodes, args.seq_len, args.dropout, args.MTGAT_alpha, args.time_gat_embed_dim, args.use_gatv2)
        self.gru = GRULayer(3 * args.num_nodes, args.MTGAT_gru_hid_dim, args.MTGAT_gru_n_layers, args.dropout)
        self.forecasting_model = Forecasting_Model(args.preq_len, args.MTGAT_gru_hid_dim,args.MTGAT_forecast_hid_dim, args.num_nodes, args.MTGAT_forecast_n_layers, args.dropout)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, batches_seen=None):
        # x shape (batch_size, seq_len, num_sensor, input_dim)
        if x.shape[-1]!=1:
            raise Exception('MTGAT Model only concern 3-d data!')
        x = torch.squeeze(x, dim=-1)

        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (B, T, 3N)

        h_end = self.gru(h_cat)           # h_end (1,batch,hidden_dim)
        h_end = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end)
        # B,T,N,C
        predictions = torch.unsqueeze(predictions, dim=-1)
        return predictions