import pickle
import numpy as np
import pandas as pd
from geopy.distance import geodesic

def process_graph_data_for_HK(data_path, save_path):
    df_raw = pd.read_csv(data_path)
    distance_matrix = []

    # 计算距离
    for i in range(len(df_raw)):
        distances = []
        for j in range(len(df_raw)):
            coords_1 = (df_raw.loc[i, '纬度'], df_raw.loc[i, '经度'])
            coords_2 = (df_raw.loc[j, '纬度'], df_raw.loc[j, '经度'])
            distance = geodesic(coords_1, coords_2).km
            distances.append(distance)
        distance_matrix.append(distances)
    distance_array = np.array(distance_matrix)
    E = np.eye(distance_array.shape[0])
    row_sums = distance_array .sum(axis=1)  # 计算每一行的总和
    distance_array  = distance_array  / row_sums[:, np.newaxis] 
    distance_array +=E
    with open(save_path, 'wb') as f:
        pickle.dump(distance_array, f)

process_graph_data_for_HK(r'dataset\Exit-and-entry\origin_data\stations_locations.csv',
                           r'dataset\Exit-and-entry\HK_stations_graph.pkl')