import numpy as np
import pandas as pd
import torch
from kan import KAN


def undersample(data):
    # 提取特征和标签
    X = data[:, :-1]  # 特征
    y = data[:, -1]  # 标签

    # 计算每个类别的样本数量
    num_class_0 = np.sum(y == 0)
    num_class_1 = np.sum(y == 1)

    # 找出较少的类别样本数量
    minority_class = np.min([num_class_0, num_class_1])

    # 随机选择较多类别中与较少类别数量相同的样本，实现欠采样
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    undersampled_indices = np.concatenate([np.random.choice(class_0_indices, minority_class, replace=False),
                                           np.random.choice(class_1_indices, minority_class, replace=False)])

    # 获取欠采样后的特征和标签
    X_undersampled = X[undersampled_indices]
    y_undersampled = y[undersampled_indices]

    # 欠采样后的数据
    undersampled_data = np.column_stack((X_undersampled, y_undersampled))

    np.random.shuffle(undersampled_data)

    return undersampled_data

if __name__ == "__main__":
    model = KAN([7, 10, 1], device="cuda")
    df = pd.read_csv("data_OK.csv")
    data = df.to_numpy().astype('float32')
    data = undersample(data)
    dataset = {}
    length = data.shape[0] // 10
    dataset['train_input'] = torch.from_numpy(data[:-length, 2:9]).to("cuda")
    dataset['test_input'] = torch.from_numpy(data[-length:, 2:9]).to("cuda")
    dataset['train_label'] = torch.from_numpy(data[:-length, 9]).reshape((-1, 1)).to("cuda")
    dataset['test_label'] = torch.from_numpy(data[-length:, 9]).reshape((-1, 1)).to("cuda")
    # print(dataset['train_input'].shape)
    # print(dataset['test_label'][0:20,:])
    # print(dataset['train_label'])



    def train_acc():
        return torch.mean((torch.round(model(dataset['train_input'])[:, 0]) == dataset['train_label'][:, 0]).float())


    def test_acc():
        return torch.mean((torch.round(model(dataset['test_input'])[:, 0]) == dataset['test_label'][:, 0]).float())



    results = model.train(dataset, opt="LBFGS", steps=20, metrics=(train_acc, test_acc), device="cuda:0")
    print(results['train_acc'][-1])
    print(results['test_acc'][-1])

    predicted = model(dataset['test_input'])[:, 0]