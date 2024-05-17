
import time
from datetime import datetime
import json
from torch.optim.lr_scheduler import StepLR

from sampled import *
from model import *
from train_net import *
from figure import plot_figure
import torch
from torch import nn

BATCH_SIZE = 256
EPOCH = 20
LR = 0.001
GAMMA = 0
STEP_SIZE = 10  # 每隔多少个 epoch 衰减一次学习率
device = "cuda"
SAMPLE_METHOD = "less"

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


def train_BiLSTM_BN(data_idx=0):
    model_name = f"BiLSTM_BN_{SAMPLE_METHOD}sample_2_{data_idx}"

    my_model = BiLSTM_BN()
    my_model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    start_time = time.time()
    info = train_val_net(model_name, EPOCH, my_model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training finished in {elapsed_time:.2f} seconds.")
    plot_figure(info, model_name)
    # 定义超参数
    hyperparameters = {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCH": EPOCH,
        "LR": LR,
        "GAMMA": GAMMA,
        "STEP_SIZE": STEP_SIZE,
        "device": device,
        "SAMPLE_METHOD": SAMPLE_METHOD
    }
    # 将超参数保存到JSON文件
    with open(f'./{model_name}/hyperparameters.json', 'w') as json_file:
        json.dump(hyperparameters, json_file, indent=4)

def train_best_BiLSTM_BN(data_idx=0):
    model_name = f"BiLSTM_BN_{SAMPLE_METHOD}sample_best_1_{data_idx}"

    my_model = BiLSTM_BN()
    my_model.to(device)
    best_model = BiLSTM_BN()
    best_model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    start_time = time.time()
    info = train_best_val_net(model_name, EPOCH, my_model, best_model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training finished in {elapsed_time:.2f} seconds.")
    plot_figure(info, model_name)
    # 定义超参数
    hyperparameters = {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCH": EPOCH,
        "LR": LR,
        "GAMMA": GAMMA,
        "STEP_SIZE": STEP_SIZE,
        "device": device,
        "SAMPLE_METHOD": SAMPLE_METHOD
    }
    # 将超参数保存到JSON文件
    with open(f'./{model_name}/hyperparameters.json', 'w') as json_file:
        json.dump(hyperparameters, json_file, indent=4)


def train_BiLSTM_BN_3layers(data_idx=0):
    model_name = f"BiLSTM_BN_3layers_best_model_{SAMPLE_METHOD}sample_{data_idx}"
    print(model_name)

    my_model = BiLSTM_BN_3layers()
    my_model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    start_time = time.time()
    info = train_val_net(model_name, EPOCH, my_model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training finished in {elapsed_time:.2f} seconds.")
    plot_figure(info, model_name)
    # 定义超参数
    hyperparameters = {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCH": EPOCH,
        "LR": LR,
        "GAMMA": GAMMA,
        "STEP_SIZE": STEP_SIZE,
        "device": device,
        "SAMPLE_METHOD": SAMPLE_METHOD
    }
    # 将超参数保存到JSON文件
    with open(f'./{model_name}/hyperparameters.json', 'w') as json_file:
        json.dump(hyperparameters, json_file, indent=4)


def train_BiLSTM_CNN_Transformer(data_idx=0):
    model_name = f"BiLSTM_CNN_Transformer_{SAMPLE_METHOD}sample_{data_idx}"
    print(model_name)

    my_model = BiLSTM_CNN_Transformer(7, 128, 2,
                                      0.5, 10, 3,
                                      1, 3, 8,
                                      4, 0.3, 3)
    my_model.to(device)
    best_model = BiLSTM_CNN_Transformer(7, 128, 2,
                                      0.5, 10, 3,
                                      1, 3, 8,
                                      4, 0.3, 3)
    best_model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    start_time = time.time()
    info = train_val_net(model_name, EPOCH, my_model, best_model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training finished in {elapsed_time:.2f} seconds.")
    plot_figure(info, model_name)
    # 定义超参数
    hyperparameters = {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCH": EPOCH,
        "LR": LR,
        "GAMMA": GAMMA,
        "STEP_SIZE": STEP_SIZE,
        "device": device,
        "SAMPLE_METHOD": SAMPLE_METHOD
    }
    # 将超参数保存到JSON文件
    with open(f'./{model_name}/hyperparameters.json', 'w') as json_file:
        json.dump(hyperparameters, json_file, indent=4)