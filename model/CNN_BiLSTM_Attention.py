import torch
import torch.nn as nn

class PredictionLayer(nn.Module):
    def __init__(self, T_dim, output_T_dim, embed_size):
        super(PredictionLayer, self).__init__()
        self.conv1 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.conv2 = nn.Conv2d(embed_size, 1, 1)

    def forward(self, input_prediction_layer):
        out = self.conv1(input_prediction_layer)
        out = out.permute(0, 3, 2, 1)
        out = self.conv2(out)
        out = out.squeeze(1)
        return out

class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, dropout, T_in, T_out):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.T_in = T_in
        self.T_out = T_out

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(dropout)
        )

        self.bi_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size * 2)

    def forward(self, x):
        B, N, T, F = x.size()
        x = x.reshape(B*N, T, F).transpose(1, 2)

        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)

        lstm_out, _ = self.bi_lstm(cnn_out)

        att_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        out = att_out.reshape(B, N, T, self.hidden_size * 2).transpose(1, 2)
        out = self.prediction_layer(out)
        return out