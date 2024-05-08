import time

from oversample import balance_data
from model import *
from train_net import train_val_net
from figure import plot_figure

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


if __name__ == '__main__':

    BATCH_SIZE = 128
    EPOCH = 10
    LR = 1e-3

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

    start_time = time.time()
    info = train_val_net("bilstm_1", EPOCH, model, train_dataloader, val_dataloader, loss_fn, optimizer)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training finished in {elapsed_time:.2f} seconds.")

    plot_figure(info)

    BiLSTM_CNN_Transformer = BiLSTM_CNN_Transformer(7, 128, 2,
                                   0.5, 10, 3,
                                   1, 3, 8,
                                   4, 0.3, 3)
    BiLSTM_CNN_Transformer.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    start_time = time.time()
    info = train_val_net("BiLSTM_CNN_Transformer", EPOCH, BiLSTM_CNN_Transformer, train_dataloader, val_dataloader, loss_fn, optimizer)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training finished in {elapsed_time:.2f} seconds.")

    plot_figure(info)