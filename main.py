from oversample import balance_data
from model import BiLSTM, LSTMClassifier
from train_net import train_val_net
from figure import plot_figure

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


if __name__ == '__main__':

    BATCH_SIZE = 512
    EPOCH = 2
    LR = 0.005

    data = torch.load('./data/data_tensor.pth')

    data_train = data['data_tensor_train']
    label_train = data['label_tensor_train']
    data_val = data['data_tensor_val']
    label_val = data['label_tensor_val']
    data_test = data['data_tensor_test']
    label_test = data['label_tensor_test']
    data = data['data_tensor_cell']
    # label = data['label_tensor_cell']

    data_train_b, label_train_b = balance_data(data_train, label_train)

    dataset_train = TensorDataset(data_train_b, label_train_b)
    dataset_val = TensorDataset(data_val, label_val)
    dataset_test = TensorDataset(data_test, label_test)

    # 利用 DataLoader 来加载数据集
    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda"
    model = BiLSTM()
    model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    info = train_val_net("bilstm_1", EPOCH, model, train_dataloader, val_dataloader, loss_fn, optimizer)
    plot_figure(info)