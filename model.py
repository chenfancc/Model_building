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


class BiLSTM_CNN_Transformer(nn.Module):
    def __init__(self, input_size_lstm, hidden_size_lstm, num_layers_lstm, dropout_rate,
                 num_filters_conv1d, kernel_size_conv1d, padding_conv1d, kernel_size_maxpool,
                 tran_feature, num_heads_tran, dropout_tran, num_layers_tran):
        super(BiLSTM_CNN_Transformer, self).__init__()
        self.input_size_lstm = input_size_lstm
        self.hidden_size_lstm = hidden_size_lstm
        self.num_layers_lstm = num_layers_lstm

        self.dropout_layer = dropout_rate

        self.num_filters_conv1d = num_filters_conv1d
        self.kernel_size_conv1d = kernel_size_conv1d
        self.padding_conv1d = padding_conv1d

        self.kernel_size_maxpool = kernel_size_maxpool

        self.bilstm = nn.LSTM(self.input_size_lstm, self.hidden_size_lstm, self.num_layers_lstm,
                              batch_first=True, bidirectional=True)
        self.conv1d = nn.Conv1d(in_channels=2 * hidden_size_lstm, out_channels=num_filters_conv1d,
                                kernel_size=kernel_size_conv1d, padding=padding_conv1d)
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size_maxpool)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=tran_feature, nhead=num_heads_tran,
                                                        dropout=dropout_tran)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers_tran)
        self.decoder = nn.Linear(tran_feature, 1)
        self.init_weights()

        self.fc = nn.Linear(in_features=1 * num_filters_conv1d, out_features=1)
        self.sig = nn.Sigmoid()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        h0 = torch.zeros(2 * self.num_layers_lstm, x.size(0), self.hidden_size_lstm).to(x.device)
        c0 = torch.zeros(2 * self.num_layers_lstm, x.size(0), self.hidden_size_lstm).to(x.device)

        lstm_out, _ = self.bilstm(x, (h0, c0))
        conv_out = torch.relu(self.conv1d(lstm_out.permute(0, 2, 1)))
        pool_out = self.maxpool(conv_out)

        mask = self._generate_square_subsequent_mask(pool_out.shape[0])
        output_tran = self.transformer_encoder(pool_out, mask)
        output_tran = self.decoder(output_tran)

        flatten_out = output_tran.flatten(1, 2)
        fc_out = self.fc(flatten_out)
        sigmoid_out = self.sig(fc_out)
        return sigmoid_out.squeeze(1).to(x.device)





if __name__ == '__main__':
    inputs = torch.ones((1280, 24, 7))
    kernel_size_maxpool = 3
    model = BiLSTM_CNN_Transformer(7, 128, 2,
                                   0.5, 10, 3,
                                   1, 3, 8,
                                   4, 0.3, 3)
    # 设置模型为 evaluation 模式
    model.eval()
    outputs = model(inputs)
