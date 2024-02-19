import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def process_origin_data_for_HK(data_path, save_path):
    df_raw = pd.read_csv(data_path)
    df_raw = df_raw[df_raw['入境 / 出境'] == '入境']
    df_raw = df_raw.drop('入境 / 出境', axis=1)
    df_raw['总计']=df_raw['总计']-df_raw['香港居民']
    # 整个数据集中屯门客运码头和红磡数据都为0，故删除此两海关口，剩余14个
    df_raw = df_raw[(df_raw['管制站'] != '红磡') & (df_raw['管制站'] != '屯门客运码头')]
    print(df_raw)
    # HK共有16个海关口，补全数据中缺失的海关口信息
    stations = df_raw['管制站'].unique()
    dates = df_raw['日期'].unique()
    new_values = np.zeros((len(dates) * len(stations), 6), dtype=object)
    for i, date in enumerate(dates):
        for j, station in enumerate(stations):
            subset = df_raw[(df_raw['日期'] == date) & (df_raw['管制站'] == station)]
            if subset.empty:
                new_values[i * len(stations) + j] = [date, station, 0, 0, 0, 0]
            else:
                new_values[i * len(stations) + j] = subset.values[0]
        print(str(i) + '/' + str(len(dates)))
    new_values_3d = new_values[:, 2:].reshape(len(dates), len(stations), -1)
    np.savez(save_path, dates=dates, stations=stations, data=new_values_3d)


def process_origin_data_for_Canada(data_path, save_path):
    df_raw = pd.read_csv(data_path, encoding='gbk')
    df_raw = df_raw[df_raw['Port of Entry'] != 'TRC']
    # 对Region和Mode进行编码
    RegionEncoder = LabelEncoder()
    ModeEncoder = LabelEncoder()
    df_raw['Region'] = RegionEncoder.fit_transform(df_raw['Region'])
    df_raw['Mode'] = ModeEncoder.fit_transform(df_raw['Mode'])
    # 获取编码映射关系
    RegionEncoderMapping = dict(zip(RegionEncoder.classes_, RegionEncoder.transform(RegionEncoder.classes_)))
    ModeEncoderMapping = dict(zip(ModeEncoder.classes_, ModeEncoder.transform(ModeEncoder.classes_)))
    ModeEncoderMapping['Unknown'] = -1
    # 补全数据中缺失的海关口信息
    dates = df_raw['Date'].unique()
    stations = df_raw[['Port of Entry', 'Region']].drop_duplicates()
    # 通过索引提速
    df_indexed = df_raw.set_index(['Date', 'Port of Entry', 'Region'])
    df_indexed = df_indexed.sort_index()

    new_values = np.zeros((len(dates) * len(stations), 5), dtype=object)
    for i, date in enumerate(dates):
        for j, station in enumerate(stations.values):
            try:
                subset = df_indexed.loc[(date, station[0], station[1])]
                new_values[i * len(stations) + j] = [date, station[0], station[1], subset.values[0][0],
                                                     subset.values[0][1]]
            except KeyError:
                new_values[i * len(stations) + j] = [date, station[0], station[1], -1, 0]
        print(str(i)+'/'+str(len(dates)))
    new_values_3d = new_values[:, 2:].reshape(len(dates), len(stations), -1)
    np.savez(save_path, dates=dates, stations=stations, data=new_values_3d,
             RegionEncoderMapping=np.array(list(RegionEncoderMapping.items())),
             ModeEncoderMapping=np.array(list(ModeEncoderMapping.items())))


#process_origin_data_for_Canada(r'..\dataset\Exit-and-entry\origin_data\open-government-traveller-report-daily-en.csv',
#                               r'..\dataset\Exit-and-entry\Canada_Daily_Arrivals.npz')

process_origin_data_for_HK(r'dataset\Exit-and-entry\origin_data\statistics_on_daily_passenger_traffic.csv',
                           r'dataset\Exit-and-entry\HK_Daily_Arrivals.npz')
