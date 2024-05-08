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