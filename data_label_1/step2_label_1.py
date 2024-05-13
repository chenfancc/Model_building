import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime


def tick_label(numpy_array, time_DIF, type):
    time_column = np.zeros(numpy_array.shape[0], dtype=float)
    for stay_id in tqdm(np.unique(numpy_array[:, 2]), desc="Processing IDs"):  # 添加进度条
        # print(id)
        first_row_index = np.where(numpy_array[:, 2] == stay_id)[0][0]
        # print(first_row_index)
        first_time = pd.Timestamp(numpy_array[first_row_index, 5])
        # print(first_time)

        rows_with_id = np.where(numpy_array[:, 2] == stay_id)[0]
        # print(rows_with_id)
        time_column[rows_with_id] = (
                                            pd.to_datetime(
                                                numpy_array[rows_with_id, 5]) - first_time).total_seconds() // 3600
        # print(time_column[rows_with_id])
        # break
    numpy_array = np.column_stack((numpy_array, time_column))

    label_column = np.zeros(numpy_array.shape[0], dtype=int)
    for idx in tqdm(range(numpy_array.shape[0]), desc="Processing rows"):  # 添加进度条
        time_diff = (pd.to_datetime(numpy_array[idx, 3]) -
                     pd.to_datetime(numpy_array[idx, 5])).total_seconds() // 3600
        if time_diff <= time_DIF:
            label_column[idx] = 1
        else:
            label_column[idx] = 0
        # break
    numpy_array = np.column_stack((numpy_array, label_column))

    numpy_array_dropped = np.delete(numpy_array, [0, 1, 3, 4, 5], axis=1)
    # print(numpy_array_dropped)
    if type == "blood":
        df = pd.DataFrame(numpy_array_dropped, columns=['stay_id', 'sbp', 'dbp', 'map', 'time', 'label'])
        print(df)
        df_mean = df.groupby(['stay_id', 'time']).mean().reset_index()
    elif type == "hr":
        df = pd.DataFrame(numpy_array_dropped, columns=['stay_id', 'hr', 'time', 'label'])
        df_mean = df.groupby(['stay_id', 'time']).mean().reset_index()
    elif type == "rr":
        df = pd.DataFrame(numpy_array_dropped, columns=['stay_id', 'rr', 'time', 'label'])
        df_mean = df.groupby(['stay_id', 'time']).mean().reset_index()
    elif type == "spo2":
        df = pd.DataFrame(numpy_array_dropped, columns=['stay_id', 'spo2', 'time', 'label'])
        df_mean = df.groupby(['stay_id', 'time']).mean().reset_index()
    elif type == "temp":
        df = pd.DataFrame(numpy_array_dropped, columns=['stay_id', 'temp', 'time', 'label'])
        df_mean = df.groupby(['stay_id', 'time']).mean().reset_index()

    return df_mean


current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
print("Start Time =", current_time)

# time.sleep(60*30)  # 暂停 10 秒

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
print("Start Time =", current_time)

data_dirs = ["../data", "../data/label_1", "../data/label_1/ticked_data"]  # 定义需要检查的目录列表

for directory in data_dirs:  # 检查并创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")

df_1 = pd.read_excel("../data/raw_data/blood_pressure_with_label.xlsx")
numpy_array_1 = df_1.to_numpy()
print("numpy_array_1 has been loaded")

df_2 = pd.read_excel("../data/raw_data/heart_rate_with_label.xlsx")
numpy_array_2 = df_2.to_numpy()
print("numpy_array_2 has been loaded")

df_3 = pd.read_excel("../data/raw_data/respiratory_rate_with_label.xlsx")
numpy_array_3 = df_3.to_numpy()
print("numpy_array_3 has been loaded")

df_4 = pd.read_excel("../data/raw_data/spo2_with_label.xlsx")
numpy_array_4 = df_4.to_numpy()
print("numpy_array_4 has been loaded")

df_5 = pd.read_excel("../data/raw_data/temperature_with_label.xlsx")
numpy_array_5 = df_5.to_numpy()
print("numpy_array_5 has been loaded")

for i in [20, 24, 30, 36, 48]:
    print(f"i = {i}")
    data_1 = tick_label(numpy_array_1, i, "blood")
    data_2 = tick_label(numpy_array_2, i, "hr")
    data_3 = tick_label(numpy_array_3, i, "rr")
    data_4 = tick_label(numpy_array_4, i, "spo2")
    data_5 = tick_label(numpy_array_5, i, "temp")

    data_1.to_csv(f"../data/label_1/ticked_data/data_1_{i}.csv", index=False)
    data_2.to_csv(f"../data/label_1/ticked_data/data_2_{i}.csv", index=False)
    data_3.to_csv(f"../data/label_1/ticked_data/data_3_{i}.csv", index=False)
    data_4.to_csv(f"../data/label_1/ticked_data/data_4_{i}.csv", index=False)
    data_5.to_csv(f"../data/label_1/ticked_data/data_5_{i}.csv", index=False)

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
print("Start Time =", current_time)
