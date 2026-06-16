import torch.nn as nn

from _Support.TemporalConvNet import TemporalConvNet
from _Support.convLSTM import ConvLSTM
from GNN.GAT import GAT_Layer
from GNN.ChebGCN import ChebGCN_Layer
from torch_geometric.nn import ChebConv
from GNN.ChebGraphConv import ChebGraphConv
from GNN.GAT import convert_adj_sparse


class PredictionLayer(nn.Module):
    def __init__(self, T_dim, output_T_dim, embed_size):
        super(PredictionLayer, self).__init__()
        self.conv1 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.conv2 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, input_prediction_layer):
        """
        :param input_prediction_layer: [B, T, N, D]
        :return: [B, N, out_T]
        """
        out = self.relu(self.conv1(input_prediction_layer))  # 等号左边 out shape: [B, T, N, d]
        out = out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, d, N, T]
        out = self.conv2(out)  # 等号左边 out shape: [B, 1, N, T]
        out = out.squeeze(1)

        return out


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_size, T_in, T_out, dropout):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: [B, N, t, d]
        :return: [B, N, T_out]
        """
        out = self.linear1(x)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out = self.linear3(out)
        out = out.permute(0, 2, 1, 3)
        out = self.prediction_layer(out)
        return out


class LSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, dropout, T_in, T_out):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x):
        B, N, t, d = x.size()
        x = x.view(B*N, t, d)
        out, _ = self.lstm(x)
        out = out.view(B, N, t, -1)
        out = out.permute(0, 2, 1, 3)
        out = self.prediction_layer(out)
        return out


class GRU(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, dropout, T_in, T_out):
        super(GRU, self).__init__()
        self.gru = nn.GRU(in_channels, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x):
        B, N, t, d = x.size()
        x = x.view(B*N, t, d)
        out, _= self.gru(x)
        out = out.view(B, N, t, -1)
        out = out.permute(0, 2, 1, 3)
        out = self.prediction_layer(out)
        return out


class TCN(nn.Module):
    def __init__(self, in_channels, num_channels, kernel_size, dropout, T_in, T_out):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=in_channels, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=num_channels[-1])

    def forward(self, x):
        B, N, t, d = x.size()
        x = x.view(B*N, t, d)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        out = out.transpose(1, 2)
        out = out.view(B, N, t, -1)
        out = out.permute(0, 2, 1, 3)
        out = self.prediction_layer(out)
        return out


class CNN_LSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, dropout, T_in, T_out):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=hidden_size, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x):
        B, N, t, d = x.size()
        x = x.view(B*N, t, d)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x = x.view(B*N, t, -1)
        out, _ = self.lstm(x)
        out = out.view(B, N, t, -1)
        out = out.permute(0, 2, 1, 3)
        out = self.prediction_layer(out)
        return out


class Conv_LSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, T_in, T_out):
        super(Conv_LSTM, self).__init__()
        self.linear1 = nn.Linear(in_features=in_channels, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=1, out_features=hidden_size)
        self.convLSTM = ConvLSTM(input_dim=hidden_size, hidden_dim=hidden_size, kernel_size=(3, 3),
                                 num_layers=num_layers, batch_first=True, bias=True, return_all_layers=False)
        self.linear3 = nn.Linear(in_features=hidden_size, out_features=1)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x):
        out = self.linear1(x)  # [B, N, t, d]
        out = out.permute(0, 2, 1, 3).unsqueeze(-1)  # [B, T, N, d, 1]
        out = out.permute(0, 1, 3, 2, 4)  # [B, T, d, N, 1]
        out, _ = self.convLSTM(out)  # 输出形状 [B, T, d, N, 1]
        out = out[-1].squeeze(-1)  # [B, T, d, N]
        out = out.permute(0, 1, 3, 2)
        out = self.prediction_layer(out)
        return out


class GCN_LSTM(nn.Module):
    def __init__(self, adj, in_channels, hidden_size, device, K, num_layers, dropout, T_in, T_out):
        super(GCN_LSTM, self).__init__()
        self.adj = adj
        self.cheb_conv = ChebGCN_Layer(device, in_channels, hidden_size, K)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x):
        B, N, t, d = x.size()
        x = x.permute(0, 2, 1, 3)
        out = self.cheb_conv(x, self.adj)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B*N, t, -1)
        out, _ = self.lstm(out)
        out = out.reshape(B, N, t, -1)
        out = out.permute(0, 2, 1, 3)
        out = self.prediction_layer(out)
        return out


class GC_LSTM(nn.Module):
    def __init__(self, adj, in_channels, hidden_size, K, num_layers, dropout, T_in, T_out):
        super(GC_LSTM, self).__init__()
        self.adj = adj
        self.gcn = ChebConv(in_channels=in_channels, out_channels=hidden_size, K=K)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x):
        edge_index, edge_attr = convert_adj_sparse(self.adj)
        B, N, t, d = x.size()
        out = x.permute(0, 2, 1, 3)
        out = out.reshape(B*t, N, -1)
        out = self.gcn(out, edge_index=edge_index, edge_weight=edge_attr)
        out = out.reshape(B, t, N, -1)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B*N, t, -1)
        out, _ = self.lstm(out)
        out = out.reshape(B, N, t, -1)
        out = out.permute(0, 2, 1, 3)
        out = self.prediction_layer(out)
        return out


class ChebGCN_LSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, K, adj, num_layers, dropout, T_in, T_out):
        super(ChebGCN_LSTM, self).__init__()
        self.gcn = ChebGraphConv(in_channels, hidden_size, K, adj, bias=True)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x):
        B, N, t, d = x.size()
        out = x.permute(0, 2, 1, 3)
        out = self.gcn(out)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B*N, t, -1)
        out, _ = self.lstm(out)
        out = out.reshape(B, N, t, -1)
        out = out.permute(0, 2, 1, 3)
        out = self.prediction_layer(out)
        return out



class GAT_LSTM(nn.Module):
    def __init__(self, adj, in_channels, hidden_size, device, num_layers, dropout, T_in, T_out):
        super(GAT_LSTM, self).__init__()
        self.adj = adj
        self.GAT = GAT_Layer(device, in_channels, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x):
        B, N, t, d = x.size()
        out = x.permute(0, 2, 1, 3)
        out = self.GAT(out, self.adj)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B*N, t, -1)
        out, _ = self.lstm(out)
        out = out.reshape(B, N, t, -1)
        out = out.permute(0, 2, 1, 3)
        out = self.prediction_layer(out)
        return out
