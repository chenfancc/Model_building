import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from simulation_data import synthetic_data
from model import BiLSTM, LSTMClassifier
from train_net import train_net, train_val_net

if __name__ == '__main__':
    data_train, label_train = synthetic_data(100, 24, 7, 0.3)
    data_test, label_test = synthetic_data(100, 24, 7, 0.3)
    data_val, label_val = synthetic_data(10, 24, 7, 0.3)


    dataset_train = TensorDataset(data_train, label_train)
    dataset_test = TensorDataset(data_test, label_test)
    dataset_val = TensorDataset(data_val, label_val)

    BATCH_SIZE = 32
    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda"
    model = BiLSTM()
    model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # train_net("test_bilstm", 100, model, train_dataloader, train_dataloader, loss_fn, optimizer, device)
    info = train_val_net("test_lstm", 20, model, train_dataloader, val_dataloader, loss_fn, optimizer)





