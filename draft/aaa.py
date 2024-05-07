def forward_print(self, x):
    lstm_out, _ = self.bilst(x)
    print(f'BiLSTM:{lstm_out.shape}')

    conv_out = self.conv1d(lstm_out.permute(0, 2, 1))
    conv_out = F.relu(conv_out)
    print(f'Conv1d:{conv_out.shape}')

    pooled_out = self.maxpool(conv_out)
    print(f'MaxPool:{pooled_out.shape}')

    attn_output, attn_weights = self.attention(pooled_out, pooled_out, pooled_out)
    print(f'Attention:{attn_output.shape}')

    flatten_out = attn_output.flatten(1, 2)
    print(f'Flatten:{flatten_out.shape}')

    fc_output = self.fc(flatten_out)
    print(f'FC:{fc_output.shape}')

    output = self.softmax(fc_output)
    print(f'Output:{output.shape}')

    return output













def __init__(self, input_size, hidden_size, num_layers, num_filters, kernel_size, padding, time_step, num_heads,
             num_classes):
    super(BiLSTM_CNN_SelfAttention, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bilstm = nn.LSTM(7, 128, 2, bidirectional=True)
    self.conv1d = nn.Conv1d(in_channels=256, out_channels=10, kernel_size=3,
                            padding=1)
    self.maxpool = nn.MaxPool1d(kernel_size=3)
    self.attention = nn.MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.5)
    self.fc = nn.Linear(80, 2)
    self.softmax = nn.Softmax(dim=1)

model = BiLSTM_CNN_SelfAttention(input_size=data_train.shape[2], hidden_size=128, num_layers=2,
                                 num_filters=10, kernel_size=3, padding=1,
                                 time_step=data_train.shape[0], num_heads=2,
                                 num_classes=2)










