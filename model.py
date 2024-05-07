import torch
from torch import nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        c0 = torch.zeros(4, x.size(0), 128).to(x.device)

        output, _ = self.lstm(x, (h0, c0))
        # print(output.shape)
        output_hd = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        output_drorout = self.dropout(output_hd)
        # print(output.shape)
        output_fc1 = torch.relu(self.fc1(output_drorout))
        # print(output.shape)
        output_fc2 = torch.relu(self.fc2(output_fc1))
        # print(output.shape)
        output_fc3 = torch.relu(self.fc3(output_fc2))
        # print(output.shape)
        output_fc4 = torch.relu(self.fc4(output_fc3))
        # print(output.shape)
        output_sigmoid = torch.sigmoid(self.fc5(output_fc4))
        # print(output.shape)
        return output_sigmoid.squeeze(1).to(x.device)


class LSTMClassifier(nn.Module):
    def __init__(self, device):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = 128
        self.num_layers = 1
        self.device = device
        self.lstm = nn.LSTM(7, 128, 1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2,
                         x.size(0),
                         self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2,
                         x.size(0),
                         self.hidden_size).to(self.device)

        # LSTM层
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 两个线性层
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out.squeeze(1).to(self.device)
class BiLSTM_CNN_SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_filters, kernel_size, padding, time_step, num_heads,
                 num_classes):
        super(BiLSTM_CNN_SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilst = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.conv1d = nn.Conv1d(in_channels=2 * hidden_size, out_channels=num_filters, kernel_size=kernel_size,
                                padding=padding)
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size)
        self.attention = nn.MultiheadAttention(embed_dim=8, num_heads=num_heads, dropout=0.5)
        self.fc = nn.Linear(8 * num_filters, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.bilst(x)

        conv_out = self.conv1d(lstm_out.permute(0, 2, 1))
        conv_out = F.relu(conv_out)

        pooled_out = self.maxpool(conv_out)

        attn_output, attn_weights = self.attention(pooled_out, pooled_out, pooled_out)

        flatten_out = attn_output.flatten(1, 2)

        fc_output = self.fc(flatten_out)

        output = self.softmax(fc_output)

        return output


if __name__ == '__main__':
    inputs = torch.ones((1280, 24, 7))
    model = BiLSTM_CNN_SelfAttention(input_size=7, hidden_size=128, num_layers=2,
                                     num_filters=10, kernel_size=3, padding=1,
                                     time_step=inputs.shape[1], num_heads=2,
                                     num_classes=2)
    # 设置模型为 evaluation 模式
    model.eval()


    # 使用 torch.jit.trace 跟踪模型
    traced_model = torch.jit.trace(model, inputs)

    # 保存跟踪后的模型
    traced_model.save('traced_model.pt')
