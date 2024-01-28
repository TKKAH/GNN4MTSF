import argparse
import os

import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.logger import get_logger, get_log_dir
from utils.print_args import log_args
import random
import numpy as np

if __name__ == '__main__':
    os.system('')
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='GNN4MTSF')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/traffic/PEMS-BAY', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='pems-bay.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--split_type', type=str, default='amount')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--scale', type=bool, default=True)
    parser.add_argument('--scale_type', type=str, default='std')
    parser.add_argument('--scale_column_wise', type=bool, default=True)
    parser.add_argument('--predefined_graph', type=bool, default=False)
    parser.add_argument('--graph_path', type=str, default='adj_mx_pems_bay.pkl')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # 下列为各模型供调的所有参数，注释的参数表示多模型共用 
    # for DCRNN 
    # predefined_graph should be true
    parser.add_argument('--num_nodes', type=int, default=325)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=32) 
    parser.add_argument('--cheb_k', default=2, type=int)
    parser.add_argument('--cl_decay_steps', type=int, default=1000)
    parser.add_argument('--use_curriculum_learning', type=bool, default=True)
    parser.add_argument('--filter_type', type=str, default='laplacian')
    # for AGCRN
    # predefined_graph should be false

    # for HiPPOAGCRN
    # predefined_graph should be false
    parser.add_argument('--HiPPOorder', type=int, default=64)

    # for MTGAT
    # predefined_graph should be false
    # parser.add_argument('--num_nodes', type=int, default=325)
    # parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    # parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--MTGAT_kernel_size', type=int, default=7)
    parser.add_argument('--MTGAT_use_gatv2', type=bool, default=True)
    parser.add_argument('--MTGAT_alpha', type=float, default=0.2)
    parser.add_argument('--MTGAT_gru_n_layers', type=int, default=1)
    parser.add_argument('--MTGAT_gru_hid_dim', type=int, default=150)
    parser.add_argument('--MTGAT_forecast_n_layers', type=int, default=1)
    parser.add_argument('--MTGAT_forecast_hid_dim', type=int, default=150)
    parser.add_argument('--MTGAT_feat_gat_embed_dim', type=int, default=None)
    parser.add_argument('--MTGAT_time_gat_embed_dim', type=int, default=None)

    #for STSGCN
    # predefined_graph should be true
    # parser.add_argument('--input_dim', type=int, default=1)
    # parser.add_argument('--num_nodes', type=int, default=325)
    # parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    # parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--STSGCN_hidden_dims', type=list, default=[[64, 64, 64], [64, 64, 64], [64, 64, 64], [64, 64, 64]])
    parser.add_argument('--STSGCN_out_layer_dim', type=int, default=128)
    parser.add_argument('--STSGCN_first_layer_embedding_size', type=int, default=64)
    parser.add_argument('--STSGCN_activation', type=str, default='GLU')
    parser.add_argument('--STSGCN_use_mask', type=bool, default=True)
    parser.add_argument('--STSGCN_strides', type=int, default=3)

    # for FCSTGNN
    # predefined_graph should be false
    # parser.add_argument('--input_dim', type=int, default=1)
    # parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    # parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # parser.add_argument('--num_nodes', type=int, default=325)
    parser.add_argument('--FCSTGNN_pooling_choice', type=str, default='mean')
    parser.add_argument('--FCSTGNN_decay', type=float, default=0.7)
    parser.add_argument('--FCSTGNN_conv_kernel', type=int, default=6)
    parser.add_argument('--FCSTGNN_moving_window', type=int, default=2)
    parser.add_argument('--FCSTGNN_1DCNN_hidden_dim', type=int, default=48)
    parser.add_argument('--FCSTGNN_1DCNN_output_dim', type=int, default=18)
    parser.add_argument('--FCSTGNN_hidden_dim', type=int, default=16)
    parser.add_argument('--FCSTGNN_out_layer_dim', type=int, default=128)

    # for GTS
    # predefined_graph can be true or false
    # parser.add_argument('--num_nodes', type=int, default=325)
    # parser.add_argument('--input_dim', type=int, default=1)
    # parser.add_argument('--hidden_dim', type=int, default=64)
    # parser.add_argument('--output_dim', type=int, default=1)
    # parser.add_argument('--num_layers', type=int, default=1)
    # parser.add_argument('--embed_dim', type=int, default=32) 
    # parser.add_argument('--cheb_k', default=2, type=int)
    # parser.add_argument('--cl_decay_steps', type=int, default=1000)
    # parser.add_argument('--use_curriculum_learning', type=bool, default=True)
    # parser.add_argument('--filter_type', type=str, default='laplacian')
    parser.add_argument('--GTS_temperature', type=float, default=0.5)
    parser.add_argument('--GTS_neighbor_graph_k', type=int, default=5)

    # for MSGNet
    # parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    # parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--MSGNet_enc_dim', type=int, default=32)
    parser.add_argument('--MSGNet_enc_layers', type=int, default=1)
    parser.add_argument('--MSGNet_Predict_Layer_individual', type=bool, default=True)
    parser.add_argument('--MSGNet_MSGNet_node_dim', type=int, default=10)
    parser.add_argument('--MSGNet_conv_channel', type=int, default=32)
    parser.add_argument('--MSGNet_skip_channel', type=int, default=32)
    parser.add_argument('--MSGNet_gcn_depth', type=int, default=2)
    parser.add_argument('--MSGNet_propalpha', type=float, default=0.3)
    parser.add_argument('--MSGNet_n_heads', type=int, default=8)
    parser.add_argument('--MSGNet_FTT_top_k', type=int, default=5)

    # for CrossGNN
    # parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    # parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--CrossGNN_e_layers', type=int, default=1)
    parser.add_argument('--CrossGNN_anti_ood', type=bool, default=True)
    parser.add_argument('--CrossGNN_FTT_top_k', type=int, default=4)
    parser.add_argument('--CrossGNN_enc_dmodel', type=int, default=8)
    parser.add_argument('--CrossGNN_nvechidden', type=int, default=1)
    parser.add_argument('--CrossGNN_tvechidden', type=int, default=1)
    parser.add_argument('--CrossGNN_use_tgcn', type=bool, default=True)
    parser.add_argument('--CrossGNN_use_ngcn', type=bool, default=True)
    parser.add_argument('--CrossGNN_tgcn_tk', type=int, default=10)
    
    # for ASTGCN
    # parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    # parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # parser.add_argument('--num_nodes', type=int, default=325)
    parser.add_argument('--ASTGCN_cheb_k', default=3, type=int)
    parser.add_argument('--ASTGCN_input_dim', default=1, type=int)
    parser.add_argument('--ASTGCN_nb_chev_out_dim', default=64, type=int)
    parser.add_argument('--ASTGCN_nb_time_out_dim', default=64, type=int)
    parser.add_argument('--ASTGCN_nb_block', default=2, type=int)
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--loss_with_regularization', type=bool, default=False)
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # logging
    log_dir = get_log_dir(args)
    logger = get_logger(log_dir, __name__, 'info.log', level='INFO')

    logger.info('Args in experiment:')
    log_args(args, logger)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        raise Exception("Invalid Task!")

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args, logger)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            logger.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args, logger)  # set experiments
        logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
