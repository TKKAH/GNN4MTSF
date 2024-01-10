import numpy as np
import torch


class NScaler(object):
    """
        No Scaler
        """

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        # 确保数据类型一致
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean


class MinMax01Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)


class MinMax11Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


class ColumnMinMaxScaler():
    # Note: to use this scale, must init the min and max with column min and column max
    def __init__(self, min, max):
        self.min = min
        self.min_max = max - self.min
        self.min_max[self.min_max == 0] = 1
        print(self.min_max)

    def transform(self, data):
        return (data - self.min) / self.min_max

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min_max = torch.from_numpy(self.min_max).to(data.device).type(torch.float32)
            self.min = torch.from_numpy(self.min).to(data.device).type(torch.float32)
        return (data * self.min_max + self.min)


def scale_dataset(train_data, data, scaler, logger, column_wise=False):
    if scaler == 'max01':
        if column_wise:
            minimum = train_data.min(axis=0, keepdims=True)
            maximum = train_data.max(axis=0, keepdims=True)
        else:
            minimum = train_data.min()
            maximum = train_data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        logger.info('Normalize the dataset by MinMax01 Normalization')
    elif scaler == 'max11':
        if column_wise:
            minimum = train_data.min(axis=0, keepdims=True)
            maximum = train_data.max(axis=0, keepdims=True)
        else:
            minimum = train_data.min()
            maximum = train_data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        logger.info('Normalize the dataset by MinMax11 Normalization')
    elif scaler == 'std':
        if column_wise:
            mean = train_data.mean(axis=0, keepdims=True)
            std = train_data.std(axis=0, keepdims=True)
        else:
            mean = train_data.mean()
            std = train_data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        logger.info('Normalize the dataset by Standard Normalization')
    elif scaler == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        logger.info('Does not normalize the dataset')
    elif scaler == 'cmax':
        # column min max, to be depressed
        # note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(train_data.min(axis=0), train_data.max(axis=0))
        data = scaler.transform(data)
        logger.info('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler
