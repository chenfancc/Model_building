import time
from datetime import datetime
from oversample import balance_data
from model import *
from train_net import train_val_net
from figure import plot_figure
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from Hyperparameters import *


def main_data_loader(data_idx):
    data = torch.load(f'./data/data_tensor_{data_idx}.pth')

    data_train = data['data_tensor_train']
    label_train = data['label_tensor_train']
    data_val = data['data_tensor_val']
    label_val = data['label_tensor_val']
    data_test = data['data_tensor_test']
    label_test = data['label_tensor_test']
    # data = data['data_tensor_cell']
    # label = data['label_tensor_cell']

    data_train_b, label_train_b = balance_data(data_train, label_train)

    dataset_train = TensorDataset(data_train_b, label_train_b)
    dataset_val = TensorDataset(data_val, label_val)
    dataset_test = TensorDataset(data_test, label_test)

    # 利用 DataLoader 来加载数据集
    train_dataloader_f = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader_f = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader_f = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader_f, val_dataloader_f, test_dataloader_f


def train_BiLSTM(data_idx=0):
    model_name = f"BiLSTM_{data_idx}"
    my_model = BiLSTM()
    my_model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=LR)

    start_time = time.time()
    info = train_val_net(model_name, EPOCH, my_model, train_dataloader, val_dataloader, loss_fn, optimizer)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training finished in {elapsed_time:.2f} seconds.")
    plot_figure(info, model_name)


def train_BiLSTM_CNN_Transformer():
    model_name = "BiLSTM_CNN_Transformer"
    my_model = BiLSTM_CNN_Transformer(7, 128, 2,
                                      0.5, 10, 3,
                                      1, 3, 8,
                                      4, 0.3, 3)
    my_model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=0.1)

    start_time = time.time()
    info = train_val_net(model_name, EPOCH, BiLSTM_CNN_Transformer, train_dataloader, val_dataloader, loss_fn,
                         optimizer)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training finished in {elapsed_time:.2f} seconds.")
    plot_figure(info, model_name)


# def train_KAN():


if __name__ == '__main__':
    for i in [4, 6, 8, 12, 24]:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
        print("Start Time =", current_time)
        train_dataloader, val_dataloader, test_dataloader = main_data_loader(i)
        train_BiLSTM(i)
        break
