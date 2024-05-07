import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from oversample import balance_data
from model import BiLSTM_CNN_SelfAttention
from train_net import train_net

BATCH_SIZE = 128
EPOCH = 10
LR = 0.005

data = torch.load('./data/data_tensor.pth')
print()
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

model = BiLSTM_CNN_SelfAttention(input_size=data_train.shape[2], hidden_size=128, num_layers=2,
                                 num_filters=10, kernel_size=3, padding=1,
                                 time_step=data_train.shape[0], num_heads=2,
                                 num_classes=2)
if torch.cuda.is_available():
    model = model.cuda()

loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=LR)


train_net("BiLSTM_CNN_SelfAttention", EPOCH, model, train_dataloader, val_dataloader, loss_fn, optimizer)

