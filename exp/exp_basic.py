import os
import torch
from torch import nn, optim

from data_provider.data_factory import data_provider
from models import Autoformer, HiPPOAGCRN, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, MTGNN, AGCRN, STWA, DCRNN
from utils.graph_load import load_graph_data
from utils.losses import mape_loss, smape_loss, mse_loss, mae_loss


class Exp_Basic(object):
    def __init__(self, args, logger):
        self.args = args
        self.model_dict = {
            'DCRNN': DCRNN,
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'MTGNN': MTGNN,
            'AGCRN': AGCRN,
            'HiPPOAGCRN':HiPPOAGCRN,
            'STWA': STWA,
        }
        self.logger = logger
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        adj_mx = None
        if self.args.predefined_graph is True:
            adj_mx = load_graph_data(os.path.join(self.args.root_path, self.args.graph_path))

        model = self.model_dict[self.args.model].Model(self.args, adj_mx, self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
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
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs

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
