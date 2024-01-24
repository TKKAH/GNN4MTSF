import os
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import torch

from utils.scale import StandardScaler

def load_graph_data(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data[2]

def create_knn_graph(root_path,data_path,k):
    if data_path.endswith(".csv"):
        df = pd.read_csv(os.path.join(root_path,data_path))
        num_samples = df.shape[0]
        num_train = round(num_samples * 0.7)
        df=df.drop(df.columns[0], axis=1)
        data = df[:num_train].values
        data = data.transpose((1, 0))
    elif data_path.endswith(".pkl"):
        np_raw = np.load(os.path.join(root_path,data_path), allow_pickle=True)
        data = np_raw['data'].astype(float)
        data = data.transpose((1, 0, 2))
        data = data.reshape((data.shape[0],-1))
    scaler = StandardScaler(mean=data.mean(), std=data.std())
    train_feas = scaler.transform(data)
    _train_feas = torch.Tensor(train_feas)
    knn_metric = 'cosine'
    g = kneighbors_graph(train_feas, k, metric=knn_metric)
    g = np.array(g.todense(), dtype=np.float32)
    return torch.Tensor(g),_train_feas

    