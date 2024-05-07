import numpy as np
import torch
from tqdm import tqdm


def synthetic_data(num_sequences, time_steps, num_features, beta=1):
    """
    0为正弦，1为直线
    形状为[num_sequences, time_steps, num_features]
    :param num_sequences:
    :param time_steps:
    :param num_features:
    :param beta:
    :return:
    """
    X = torch.zeros(num_sequences, time_steps, num_features)
    Y = torch.zeros(num_sequences, )
    generate_type = 0
    for i in tqdm(range(num_sequences // 2), desc="Generating sine sequences"):  # 添加进度条
        Y[i] = generate_type
        for k in range(num_features):
            for j in range(time_steps):
                X[i, j, k] = np.sin(j) - np.random.random() * beta
    generate_type = 1
    for i in tqdm(range(num_sequences // 2, num_sequences), desc="Generating linear sequences"):  # 添加进度条
        Y[i] = generate_type
        for k in range(num_features):
            for j in range(time_steps):
                X[i, j, k] = -1 + np.random.random() * beta
    return X, Y
