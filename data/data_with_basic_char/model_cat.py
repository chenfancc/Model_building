import torch
from torch import nn


class BiLSTM_BN(nn.Module):
    def __init__(self):
        super(BiLSTM_BN, self).__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(2, 8)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(256+8, 128)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(128)  # 批标准化层
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(64)  # 批标准化层
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(32)  # 批标准化层
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(32, 8)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(8)  # 批标准化层
        self.dropout4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(8, 1)

    def forward(self, x, y):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        c0 = torch.zeros(4, x.size(0), 128).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.dropout(output)

        output_extra = self.fc(y)
        output_extra = self.relu(output_extra)
        output = torch.concatenate((output, output_extra), dim=1)

        output = self.fc1(output)
        output = self.relu1(output)
        output = self.bn1(output)
        output = self.dropout1(output)

        output = self.fc2(output)
        output = self.relu2(output)
        output = self.bn2(output)
        output = self.dropout2(output)

        output = self.fc3(output)
        output = self.relu3(output)
        output = self.bn3(output)
        output = self.dropout3(output)

        output = self.fc4(output)
        output = self.relu4(output)
        output = self.bn4(output)
        output = self.dropout4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)

model = BiLSTM_BN()
input_1 = torch.rand((2, 24, 7))
print(input_1)
input_2 = torch.rand((2, 2))
print(input_2)
output = model(input_1, input_2)
print(output)