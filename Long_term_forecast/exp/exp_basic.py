import os
import torch
from torch import nn, optim

from data_provider.data_factory import data_provider
from models import ASTGCN, FCSTGNN, GTS, HHAGCRN, MTGAT, MTGNN, STSGCN, CrossGNN, HHAGCRNwithoutAGL, HHAGCRNwithoutHiPPO, HHAGCRNwithoutNPW,  MSGNet, AGCRN, DCRNN
from utils.graph_load import create_knn_graph, get_node_fea, load_graph_data
from utils.losses import mape_loss, smape_loss, mse_loss, mae_loss
from utils.print_args import get_parameter_number


class Exp_Basic(object):
    def __init__(self, args, logger):
        self.args = args
        self.model_dict = {
            'DCRNN': DCRNN,
            'AGCRN': AGCRN,
            'MTGAT':MTGAT,
            'STSGCN':STSGCN,
            'FCSTGNN':FCSTGNN,
            'GTS':GTS,
            'MSGNet':MSGNet,
            'CrossGNN':CrossGNN,
            'ASTGCN':ASTGCN,
            'MTGNN':MTGNN,
            'HHAGCRN':HHAGCRN,
            'HHAGCRNwithoutHiPPO':HHAGCRNwithoutHiPPO,
            'HHAGCRNwithoutNPW':HHAGCRNwithoutNPW,
            'HHAGCRNwithoutAGL':HHAGCRNwithoutAGL
        }
        self.adj_mx=None
        self.logger = logger
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        

    def _build_model(self):
        adj_mx=None
        if self.args.predefined_graph is True and (self.args.model=='GTS' or self.args.model=='HHAGCRN'):
            adj_mx = load_graph_data(os.path.join(self.args.root_path, self.args.graph_path))
            self.adj_mx=adj_mx
            node_fea=get_node_fea(self.args.root_path,self.args.data_path)
            adj_mx=(adj_mx,node_fea)
        elif self.args.predefined_graph is True:
            adj_mx = load_graph_data(os.path.join(self.args.root_path, self.args.graph_path))
            self.adj_mx=adj_mx
        elif self.args.predefined_graph is False and (self.args.model=='GTS' or self.args.model=='HHAGCRN'):
            adj_mx=[None,None]
            #adj_mx=  create_knn_graph(self.args.root_path,self.args.data_path,self.args.GTS_neighbor_graph_k)
            self.adj_mx=adj_mx[0]
        model = self.model_dict[self.args.model].Model(self.args, adj_mx, self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model.cuda(), device_ids=self.args.device_ids)
        param=get_parameter_number(model)
        self.logger.info("Param Number:"+str(param))
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            self.logger.info('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            self.logger.info('Use CPU')
        return device

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.logger)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _generate_outputs(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs
    def _spllit_outputs_and_calculate_regularization_loss(self,outputs):
        loss=0
        if self.args.model=='GTS' and self.args.loss_with_regularization is True :
            pred = outputs[1].view(outputs[1].shape[0] * outputs[1].shape[1])
            true_label = self.adj_mx.view(outputs[1].shape[0] * outputs[1].shape[1]).to(self.device)
            compute_loss = torch.nn.BCELoss()
            loss = compute_loss(pred, true_label)
            return outputs[0],loss
        elif self.args.model=='GTS' and self.args.loss_with_regularization is False:
            return outputs[0],loss
        elif self.args.model.startswith('HHAGCRN'):
            if self.args.loss_with_kl is True:
                mu=outputs[2]
                logvar=outputs[3]
                kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss+=kl_loss(mu,logvar)
            if self.args.loss_with_regularization is True:
                pred = outputs[1].view(outputs[1].shape[0] * outputs[1].shape[1])
                true_label = self.adj_mx.view(outputs[1].shape[0] * outputs[1].shape[1]).to(self.device)
                compute_loss = torch.nn.BCELoss()
                graph_loss = compute_loss(pred, true_label)
                loss+=graph_loss
            return outputs[0],loss
        else:
            return outputs,loss

    @staticmethod
    def _select_criterion(loss_name):
        if loss_name == 'MSE':
            return mse_loss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MAE':
            return mae_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def vali(self, vali_data, vali_loader, criterion):
        pass

    def train(self, setting):
        pass

    def test(self, setting, test=0):
        pass
