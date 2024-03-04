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
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='GNN4MTSF')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='DCRNN',
                        help='model name, options: [DCRNN,AGCRN,...]')
    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, required=True,help='root path of the data file')
    parser.add_argument('--data_path', type=str, required=True,help='data file')
    parser.add_argument('--freq', required=True,type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--split_type', type=str, default='amount',help='split dataset type')
    parser.add_argument('--train_ratio', type=float, default=0.7,help='train_ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2,help='test_ratio')
    parser.add_argument('--scale', type=bool, default=True,help='scale data')
    parser.add_argument('--scale_type', type=str, default='cmax',help='scale type')
    parser.add_argument('--scale_column_wise', type=bool, default=True,help='scale_column_wise')
    parser.add_argument('--predefined_graph',action='store_true',default=False,help='input graph or not')
    parser.add_argument('--graph_path', type=str, required=True,default=None,help='the graph adj path')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    
    # common forecasting task
    parser.add_argument('--seq_len', type=int, required=True, help='input sequence length')
    parser.add_argument('--pred_len', type=int, required=True, help='prediction sequence length')
    parser.add_argument('--num_nodes', type=int, required=True,help='time series number')
    parser.add_argument('--inverse', type=bool, help='inverse output data', default=False)
    parser.add_argument('--input_dim', type=int, required=True,help='the input dim of one nodes in ont timestamp')
    parser.add_argument('--output_dim', type=int, default=1,help='must be one')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout,attention in different model use different')

    # 下列为各模型供调的模型参数
    # for DCRNN 
    # predefined_graph should be true
    parser.add_argument('--DCRNN_hidden_dim', type=int, default=64)
    parser.add_argument('--DCRNN_num_layers', type=int, default=1)
    parser.add_argument('--DCRNN_embed_dim', type=int, default=32) 
    parser.add_argument('--DCRNN_cheb_k', default=2, type=int)
    parser.add_argument('--DCRNN_cl_decay_steps', type=int, default=1000)
    parser.add_argument('--DCRNN_use_curriculum_learning', type=bool, default=True)
    parser.add_argument('--DCRNN_filter_type', type=str, default='laplacian')

    # for AGCRN
    # predefined_graph should be false
    parser.add_argument('--AGCRN_hidden_dim', type=int, default=64)
    parser.add_argument('--AGCRN_num_layers', type=int, default=1)
    parser.add_argument('--AGCRN_embed_dim', type=int, default=32) 
    parser.add_argument('--AGCRN_cheb_k', default=3, type=int)

    # for HiPPOAGCRN
    # predefined_graph should be false
    parser.add_argument('--HiPPOorder', type=int, default=64)
    parser.add_argument('--HiPPOAGCRN_hidden_dim', type=int, default=64)
    parser.add_argument('--HiPPOAGCRN_num_layers', type=int, default=1)
    parser.add_argument('--HiPPOAGCRN_embed_dim', type=int, default=32) 
    parser.add_argument('--HiPPOAGCRN_cheb_k', default=2, type=int)

    # for HHAGCRN
    # predefined_graph should be false
    parser.add_argument('--HHorder', type=int, default=64)
    parser.add_argument('--HHAGCRN_hidden_dim', type=int, default=64)
    parser.add_argument('--HHAGCRN_num_layers', type=int, default=1)
    parser.add_argument('--HHAGCRN_embed_dim', type=int, default=32) 
    parser.add_argument('--HHAGCRN_cheb_k', default=3, type=int)
    parser.add_argument('--HHAGCRN_graph_gru_hidden_size', default=64, type=int)
    


    # for MTGAT
    # predefined_graph should be false
    # args.input_dim must be one
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
    parser.add_argument('--STSGCN_hidden_dims', type=str, default="[[64, 64, 64], [64, 64, 64], [64, 64, 64], [64, 64, 64]]")
    parser.add_argument('--STSGCN_out_layer_dim', type=int, default=128)
    parser.add_argument('--STSGCN_first_layer_embedding_size', type=int, default=64)
    parser.add_argument('--STSGCN_activation', type=str, default='GLU')
    parser.add_argument('--STSGCN_use_mask', type=bool, default=True)
    parser.add_argument('--STSGCN_strides', type=int, default=3)

    # for FCSTGNN
    # predefined_graph should be false
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
    parser.add_argument('--GTS_hidden_dim', type=int, default=64)
    parser.add_argument('--GTS_num_layers', type=int, default=1)
    parser.add_argument('--GTS_cheb_k', default=2, type=int)
    parser.add_argument('--GTS_cl_decay_steps', type=int, default=1000)
    parser.add_argument('--GTS_use_curriculum_learning', type=bool, default=True)
    parser.add_argument('--GTS_filter_type', type=str, default='laplacian')
    parser.add_argument('--GTS_temperature', type=float, default=0.5)
    parser.add_argument('--GTS_neighbor_graph_k', type=int, default=5)

    # for MSGNet
    # predefined_graph should be false
    # args.input_dim must be one
    parser.add_argument('--MSGNet_enc_dim', type=int, default=32)
    parser.add_argument('--MSGNet_enc_layers', type=int, default=1)
    parser.add_argument('--MSGNet_Predict_Layer_individual', type=bool, default=True)
    parser.add_argument('--MSGNet_node_dim', type=int, default=10)
    parser.add_argument('--MSGNet_conv_channel', type=int, default=32)
    parser.add_argument('--MSGNet_skip_channel', type=int, default=32)
    parser.add_argument('--MSGNet_gcn_depth', type=int, default=2)
    parser.add_argument('--MSGNet_propalpha', type=float, default=0.3)
    parser.add_argument('--MSGNet_n_heads', type=int, default=8)
    parser.add_argument('--MSGNet_FTT_top_k', type=int, default=5)

    # for CrossGNN
    # predefined_graph should be false
    # args.input_dim must be one
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
    # predefined_graph should be true
    parser.add_argument('--ASTGCN_cheb_k', default=3, type=int)
    parser.add_argument('--ASTGCN_nb_chev_out_dim', default=64, type=int)
    parser.add_argument('--ASTGCN_nb_time_out_dim', default=64, type=int)
    parser.add_argument('--ASTGCN_nb_block', default=2, type=int)

    # for MTGNN
    # predefined_graph should be false
    parser.add_argument('--MTGNN_top_k_graph', type=int, default=2)
    parser.add_argument('--MTGNN_node_embedding_dim', type=int, default=40)
    parser.add_argument('--MTGNN_residual_channels', type=int, default=32)
    parser.add_argument('--MTGNN_skip_channels', type=int, default=64)
    parser.add_argument('--MTGNN_conv_channels', type=int, default=32)
    parser.add_argument('--MTGNN_layers', type=int, default=3)
    parser.add_argument('--MTGNN_propalpha', type=float, default=0.05)
    parser.add_argument('--MTGNN_gcn_depth', type=int, default=2)
    parser.add_argument('--MTGNN_dilation_exponential', type=int, default=1)
    parser.add_argument('--MTGNN_end_channels', type=int, default=128)

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MAE', help='loss function')
    parser.add_argument('--loss_with_kl', action='store_true',default=False,help='only use for HHAGCRN model')
    parser.add_argument('--loss_with_regularization', action='store_true',default=False,help='only use for GTS and HHAGCRN model')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

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
            setting = '{}_{}_{}_{}_inputdim{}_sl{}_pl{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.input_dim,
                args.seq_len,
                args.pred_len,
                ii)

            logger.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_inputdim{}_sl{}_pl{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.input_dim,
                args.seq_len,
                args.pred_len,
                 ii)

        exp = Exp(args, logger)  # set experiments
        logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
