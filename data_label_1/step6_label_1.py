from datetime import datetime

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

import torch
import numpy as np
from tqdm import tqdm


def build_tensor_tqdm_gpt(data):
    unique_ids = np.unique(data[:, 0])
    print(unique_ids.shape)

    data_list = []  # 临时存储列表
    label_list = []  # 临时标签列表
    for i, id in enumerate(tqdm(unique_ids)):  # 添加 tqdm 进度条
        patient_data = data[data[:, 0] == id][:, 2:2 + NUM_FEATURES].astype(float)  # 获取特征A和特征B的值
        label_data = data[data[:, 0] == id][:, -1].astype(float)
        for j in range(patient_data.shape[0] - TIME_STEPS + 1):
            patient_meta_tensor = torch.tensor(patient_data[j:j + TIME_STEPS, :]).unsqueeze(0)  # 直接使用torch.tensor
            data_list.append(patient_meta_tensor)

            label_meta_tensor = torch.tensor(label_data[j + TIME_STEPS - 1]).unsqueeze(0)  # 直接使用torch.tensor
            label_list.append(label_meta_tensor)

    # 将列表转换为张量
    data_tensor = torch.cat(data_list, dim=0)
    label_tensor = torch.cat(label_list, dim=0)

    return data_tensor, label_tensor


def splited_tensor(data, train_rate, val_rate):
    unique_ids = np.unique(data[:, 0])

    train_size = int(train_rate * len(unique_ids))
    val_size = int(val_rate * len(unique_ids))

    train_id = unique_ids[:train_size]
    val_id = unique_ids[train_size:train_size + val_size]
    test_id = unique_ids[train_size + val_size:len(unique_ids)]

    train_data = np.array([row for row in data if row[0] in train_id])
    val_data = np.array([row for row in data if row[0] in val_id])
    test_data = np.array([row for row in data if row[0] in test_id])

    data_tensor_train, label_tensor_train = build_tensor_tqdm_gpt(train_data)
    data_tensor_val, label_tensor_val = build_tensor_tqdm_gpt(val_data)
    data_tensor_test, label_tensor_test = build_tensor_tqdm_gpt(test_data)

    return data_tensor_train, label_tensor_train, data_tensor_val, label_tensor_val, data_tensor_test, label_tensor_test


def run_step6_label_1():
    # 预测时间步：24h
    TIME_STEPS = 24
    # 通道数：
    NUM_FEATURES = 7
    # i：结果时间步
    for i in [20, 24, 30, 36, 48]:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
        print("Start Time =", current_time)
        data_row = pd.read_csv(f"../data/label_1/pure_data/pure_data_{i}.csv")
        data = data_row.to_numpy()

        data_tensor_train, label_tensor_train, data_tensor_val, label_tensor_val, data_tensor_test, label_tensor_test = splited_tensor(
            data, 0.8, 0.1)

        data_tensor, label_tensor = build_tensor_tqdm_gpt(data)

        file = f"data_tensor_{i}.pth"
        torch.save({'data_tensor_train': data_tensor_train,
                    'label_tensor_train': label_tensor_train,
                    'data_tensor_val': data_tensor_val,
                    'label_tensor_val': label_tensor_val,
                    'data_tensor_test': data_tensor_test,
                    'label_tensor_test': label_tensor_test,
                    'data_tensor_cell': data_tensor,
                    'label_tensor_cell': label_tensor}, file)
