import os
import warnings

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.scale import scale_dataset
from utils.split_border import split_data_border
from utils.timefeatures import time_features_encode, time_features_no_encode

warnings.filterwarnings('ignore')


class MTS_Dataset(Dataset):
    def __init__(self, root_path, data_path, split_type, train_ratio, test_ratio, freq, timeenc, flag, size,
                 features, target, scale, scale_type, scale_column_wise):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.scale_type = scale_type
        self.scale_column_wise = scale_column_wise
        self.timeenc = timeenc
        self.freq = freq
        self.split_type = split_type
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s, border2s = split_data_border(len(df_raw), self.seq_len, self.split_type, self.train_ratio,
                                               self.test_ratio)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        df_data = None
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            #  [[]] be Series not DataFrame
            df_data = df_raw[[self.target]]
        # Normalize the overall data based on the training set data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            data, self.scaler = scale_dataset(train_data.values, df_data.values, self.scale_type,
                                              self.scale_column_wise)
        else:
            data = df_data.values
        # Add time feature
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = None
        if self.timeenc == 0:
            data_stamp = time_features_no_encode(df_stamp, self.freq)
        elif self.timeenc == 1:
            data_stamp = time_features_encode(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class MSTS_Dataset(Dataset):
    def __init__(self, root_path, data_path, split_type, train_ratio, test_ratio, freq, timeenc, flag, size,
                 features, target, scale, scale_type, scale_column_wise):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.scale_type = scale_type
        self.scale_column_wise = scale_column_wise
        self.scaler = None
        self.timeenc = timeenc
        self.freq = freq
        self.split_type = split_type
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        np_raw = np.load(os.path.join(self.root_path,
                                      self.data_path), allow_pickle=True)
        data = np_raw['data'].astype(float)
        border1s, border2s = split_data_border(len(data), self.seq_len, self.split_type, self.train_ratio,
                                               self.test_ratio)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # Normalize the overall data based on the training set data
        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            data, self.scaler = scale_dataset(train_data, data, self.scale_type,
                                              self.scale_column_wise)
        # Add time feature
        df_stamp = pd.DataFrame(np_raw['dates'][border1:border2])
        df_stamp.columns = df_stamp.columns.astype(str)
        df_stamp = df_stamp.rename(columns={'0': 'date'})
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = None
        if self.timeenc == 0:
            data_stamp = time_features_no_encode(df_stamp, self.freq)
        elif self.timeenc == 1:
            data_stamp = time_features_encode(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
