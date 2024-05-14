import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def horizon_fill(data):
    # 缺失值填充函数
    def fill_missing_values(row):
        # stay_id
        row.iloc[0] = row.iloc[[0, 6, 10, 14, 18]].max()
        row.iloc[6] = row.iloc[[0, 6, 10, 14, 18]].max()
        row.iloc[10] = row.iloc[[0, 6, 10, 14, 18]].max()
        row.iloc[14] = row.iloc[[0, 6, 10, 14, 18]].max()
        row.iloc[18] = row.iloc[[0, 6, 10, 14, 18]].max()

        # time
        row.iloc[1] = row.iloc[[1, 7, 11, 15, 19]].max()
        row.iloc[7] = row.iloc[[1, 7, 11, 15, 19]].max()
        row.iloc[11] = row.iloc[[1, 7, 11, 15, 19]].max()
        row.iloc[15] = row.iloc[[1, 7, 11, 15, 19]].max()
        row.iloc[19] = row.iloc[[1, 7, 11, 15, 19]].max()

        # label
        row.iloc[5] = row.iloc[[5, 9, 13, 17, 21]].max()
        row.iloc[9] = row.iloc[[5, 9, 13, 17, 21]].max()
        row.iloc[13] = row.iloc[[5, 9, 13, 17, 21]].max()
        row.iloc[17] = row.iloc[[5, 9, 13, 17, 21]].max()
        row.iloc[21] = row.iloc[[5, 9, 13, 17, 21]].max()

        return row

    data_filled = data.apply(fill_missing_values, axis=1)

    # 排序数据
    df_sorted = data_filled.sort_values(by=[data.columns[0], data.columns[1]])

    return df_sorted


def drop_row(data):
    dropped_data = np.delete(data, [5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], axis=1)
    return dropped_data


def fill_isnan(data):
    """缺失值按照ID。先向后填充，再向前填充"""
    columns = ['id', 'time', 'sbp', 'dbp', 'map', 'hr', 'rr', 'spo2', 'temp', 'label']
    dropped_data = pd.DataFrame(data, columns=columns)

    filled_df = dropped_data.groupby('id')[['time', 'sbp', 'dbp', 'map', 'hr', 'rr', 'spo2', 'temp', 'label']].apply(
        lambda x: x.ffill().bfill())

    """剩余的缺失值按照列填充平均值"""
    column_means = filled_df.mean()
    filled_data = filled_df.fillna(column_means)
    filled_data = pd.DataFrame(filled_data.to_numpy())
    dropped_data.iloc[:, 1:7] = filled_data.iloc[:, 0:6].values
    return dropped_data


def replace_abnormal(data):
    columns = ['id', 'time', 'sbp', 'dbp', 'map', 'hr', 'rr', 'spo2', 'temp', 'label']
    df_drop = pd.DataFrame(data, columns=columns)
    feature_cols = ['sbp', 'dbp', 'map', 'hr', 'rr', 'spo2', 'temp']

    # 定义异常值的上下限
    lower_limit = {'sbp': 30, 'dbp': 30, 'map': 30, 'hr': 5, 'rr': 3, 'spo2': 5, 'temp': 15}
    upper_limit = {'sbp': 300, 'dbp': 300, 'map': 300, 'hr': 300, 'rr': 60, 'spo2': 100, 'temp': 45}

    # 处理异常值
    for col in feature_cols:
        if lower_limit[col] is not None:
            # 将小于下限的值替换为下限值
            df_drop.loc[df_drop[col] < lower_limit[col], col] = np.nan
        if upper_limit[col] is not None:
            # 将大于上限的值替换为上限值
            df_drop.loc[df_drop[col] > upper_limit[col], col] = np.nan

    df_drop['label'] = np.floor(df_drop['label'])

    df_drop.sort_values(by=['id', 'time'], inplace=True)

    return df_drop


def normalize_data(data):
    columns = ['id', 'time', 'sbp', 'dbp', 'map', 'hr', 'rr', 'spo2', 'temp', 'label']
    data_pd = pd.DataFrame(data, columns=columns)
    feature_cols = ['sbp', 'dbp', 'map', 'hr', 'rr', 'spo2', 'temp']
    # 初始化 MinMaxScaler 对象
    scaler = StandardScaler()

    # 只对特征列进行拟合和转换
    normalized_features = scaler.fit_transform(data_pd[feature_cols])

    # 将归一化后的特征值添加回原始的 DataFrame
    data_pd[feature_cols] = normalized_features

    data_pd.fillna(0, inplace=True)

    # 转换为np数组并返回
    return data_pd

def run_step5_label_1():

    start_time = time.time()
    time_list = []
    data_dirs = ["../data", "../data/label_1", "../data/label_1/pure_data/"]  # 定义需要检查的目录列表

    for directory in data_dirs:  # 检查并创建目录
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")

    for i in tqdm([20, 24, 30, 36, 48]):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
        print("Start Time =", current_time)
        data_row = pd.read_csv(f"../data/label_1/merged_data/merged_data_{i}.csv")
        data_1 = horizon_fill(data_row)
        print("horizon_fill has completed")
        data_2 = drop_row(data_1)
        print("drop_row has completed")
        data_3 = replace_abnormal(data_2)
        print("replace_abnormal has completed")
        data_4 = fill_isnan(data_3)
        print("fill_isnan has completed")
        data_5 = normalize_data(data_4)
        print("normalize_data has completed")
        np.savetxt(f"../data/label_1/pure_data/pure_data_{i}.csv", data_5, delimiter=",", fmt="%s")
        print(f"pure_data_{i} has saved")
        time_list.append(time.time() - start_time)

    print(time_list)
