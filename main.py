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
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 256
EPOCH = 20
LR = 0.001
GAMMA = 0
STEP_SIZE = 10  # 每隔多少个 epoch 衰减一次学习率
device = "cuda"
SAMPLE_METHOD = "less"


def main_data_loader(data_idx, sample_method, out_use=False):
    if out_use:
        data = torch.load(f'E:\deeplearning\Model_building\data_label_1/data_tensor_{data_idx}.pth')
    else:
        data = torch.load(f'data_label_1/data_tensor_{data_idx}.pth')

    data_train = data['data_tensor_train']
    label_train = data['label_tensor_train']
    data_val = data['data_tensor_val']
    label_val = data['label_tensor_val']
    data_test = data['data_tensor_test']
    label_test = data['label_tensor_test']
    # data_label_1 = data_label_1['data_tensor_cell']
    # label = data_label_1['label_tensor_cell']

    if sample_method == "less":
        data_train_b, label_train_b = less_balance_data(data_train, label_train)
    elif sample_method == "over":
        data_train_b, label_train_b = over_balance_data(data_train, label_train)

    dataset_train = TensorDataset(data_train_b, label_train_b)
    dataset_val = TensorDataset(data_val, label_val)
    dataset_test = TensorDataset(data_test, label_test)

    # 利用 DataLoader 来加载数据集
    train_dataloader_f = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader_f = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader_f = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader_f, val_dataloader_f, test_dataloader_f


def train_model_best(model_name, model, loss=nn.BCELoss, optimizer = torch.optim.Adam, scheduler = StepLR, best=True):
    my_model = model().to(device)
    if best: best_model = model().to(device)
    start_time = time.time()
    loss_fn = loss()
    optimizer_fn = optimizer(my_model.parameters(), lr=LR)
    scheduler_fn = scheduler(optimizer_fn, step_size=STEP_SIZE, gamma=GAMMA)
    if best:
        info = train_best_val_net(model_name, EPOCH, my_model, best_model, train_dataloader, val_dataloader, loss_fn, optimizer_fn,
                         scheduler_fn)
    else:
        info = train_val_net(model_name, EPOCH, my_model, train_dataloader, val_dataloader, loss_fn, optimizer_fn,
                         scheduler_fn)
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


if __name__ == '__main__':
    for i in [20, 24, 30, 36, 48]:
        print(f"--------------------------------------------------------------------"
              f"Now i = {i}"
              f"--------------------------------------------------------------------")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
        print("Start Time =", current_time)
        train_dataloader, val_dataloader, test_dataloader = main_data_loader(i, SAMPLE_METHOD)
        print(EPOCH)
        print(device)
        print(LR)
        print(BATCH_SIZE)
        print(GAMMA)
        print(STEP_SIZE)
        train_model_best(f"BiLSTM_BN_3layers_best_model_{SAMPLE_METHOD}sample_{i}", BiLSTM_BN_3layers)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
        print("Start Time =", current_time)
