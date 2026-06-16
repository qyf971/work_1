import torch.nn as nn

from _Support.EfficientShrinkageTemporalConvNet import TemporalConvNet as ESTCN
from _Support.TemporalConvNet import TemporalConvNet
from _Support.TransformerEncoder import TransformerEncoder
from _Support.embed import DataEmbedding
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, predict_len):
        super(MLP, self).__init__()
        self.predict_len = predict_len
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output = self.linear1(x)
        output = self.relu1(output)
        output = self.linear2(output)
        output = self.relu2(output)
        output = self.linear3(output)
        return output[:, -self.predict_len:, :]


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, predict_len):
        super(RNN, self).__init__()
        self.predict_len = predict_len
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.linear(output)
        return output[:, -self.predict_len:, :]


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, predict_len):
        super(BiRNN, self).__init__()
        self.predict_len = predict_len
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.linear(output)
        return output[:, -self.predict_len:, :]


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, predict_len):
        super(GRU, self).__init__()
        self.predict_len = predict_len
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        output = self.linear(output)
        return output[:, -self.predict_len:, :]


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, predict_len):
        super(BiGRU, self).__init__()
        self.predict_len = predict_len
        self.BiGRU = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.BiGRU(x)
        out = self.linear(out)
        return out[:, -self.predict_len:, :]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, predict_len):
        super(LSTM, self).__init__()
        self.predict_len = predict_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.linear(output)
        return output[:, -self.predict_len:, :]


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, predict_len):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_len = predict_len
        self.biLSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        output, _ = self.biLSTM(x)
        output = self.linear(output)
        return output[:, -self.predict_len:, :]


class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, predict_len):
        super(CNN, self).__init__()
        self.predict_len = predict_len
        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        return x[:, -self.predict_len:, :]



class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, predict_len):
        super(CNN_LSTM, self).__init__()
        self.predict_len = predict_len
        self.ConvLayer = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.ConvLayer(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.linear(x[:, -self.predict_len:, :])


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, predict_len):
        super(CNN_BiLSTM, self).__init__()
        self.predict_len = predict_len
        self.ConvLayer = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.ConvLayer(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.linear(x[:, -self.predict_len:, :])


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, predict_len):
        super(TCN, self).__init__()
        self.predict_len = predict_len
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output[:, -self.predict_len:, :])
        return output


class ESTCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, predict_len):
        super(ESTCN, self).__init__()
        self.predict_len = predict_len
        self.estcn = ESTCN(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        output = self.estcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output)
        return output[:, -self.predict_len:, :]


class Attention_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, d_model, num_heads, dropout, predict_len):
        super(Attention_LSTM, self).__init__()
        self.predict_len = predict_len
        self.embedding = DataEmbedding(input_size, d_model, dropout)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None):
        output = self.embedding(x)
        output, _ = self.attention(output, output, output, attn_mask=mask)
        output, _ = self.lstm(output)
        output = self.linear(output)
        return output[:, -self.predict_len:, :]


class CNN_LSTM_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, d_model, num_heads, dropout, predict_len):
        super(CNN_LSTM_Attention, self).__init__()
        self.predict_len = predict_len
        self.convLayer = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.embedding = DataEmbedding(hidden_size, d_model, dropout)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        output = self.convLayer(x)
        output = output.transpose(1, 2)
        output, _ = self.lstm(output)
        output = self.embedding(output)
        output, _ = self.attention(output, output, output, attn_mask=mask)
        output = self.linear(output)
        return output[:, -self.predict_len:, :]


class LSTM_Transformer(nn.Module):
    def __init__(self, input_size, d_model, num_heads, dropout, batch_first, lstm_hidden_size, lstm_num_layers,
                 encoderlayer_num_layers, predict_len):
        super(LSTM_Transformer, self).__init__()
        self.predict_len = predict_len

        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size

        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.enc_embedding = DataEmbedding(lstm_hidden_size, d_model)
        self.transformer_encoder = TransformerEncoder(d_model, num_heads, dropout, batch_first, encoderlayer_num_layers,
                                                      dim_feedforward=2048)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.enc_embedding(output)
        output = self.transformer_encoder(output)
        output = self.linear(output)
        return output[:, -self.predict_len:, :]


class Transformer_LSTM(nn.Module):
    def __init__(self, input_size, d_model, num_heads, lstm_hidden_size, lstm_num_layers, num_encoder_layers, dropout,
                 batch_first, predict_len):
        super(Transformer_LSTM, self).__init__()

        self.predict_len = predict_len
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size

        self.enc_embedding = DataEmbedding(input_size, d_model)
        self.transformer_encoder = TransformerEncoder(d_model, num_heads, dropout, batch_first, num_encoder_layers)
        self.lstm = nn.LSTM(d_model, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        output = self.enc_embedding(x)
        output = self.transformer_encoder(output)
        output, _ = self.lstm(output)
        output = self.linear(output)
        return output[:, -self.predict_len:, :]
