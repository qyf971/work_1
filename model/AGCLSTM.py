import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionLayer(nn.Module):
    def __init__(self, T_dim, output_T_dim, embed_size):
        super(PredictionLayer, self).__init__()
        self.conv1 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.conv2 = nn.Conv2d(embed_size, 1, 1)

    def forward(self, input_prediction_layer):
        """
        :param input_prediction_layer: [B, T, N, D]
        :return: [B, N, out_T]
        """
        out = self.conv1(input_prediction_layer)  # 等号左边 out shape: [B, T, N, d]
        out = out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, d, N, T]
        out = self.conv2(out)  # 等号左边 out shape: [B, 1, N, T]
        out = out.squeeze(1)

        return out

class AdjacencyMatrixConstructor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, xt):
        # xt: [B, N, D]
        h = self.dense(xt)  # [B, N, out_dim]
        sim = torch.matmul(h, h.transpose(-1, -2))  # [B, N, N]
        sim_relu = F.relu(sim)
        attn = F.softmax(sim_relu, dim=-1)  # 每一行归一化
        adj = (attn + attn.transpose(-1, -2)) / 2  # 对称处理
        return adj  # [B, N, N]

class AdaptiveGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.adj_constructor = AdjacencyMatrixConstructor(input_dim, hidden_dim)

        self.gcn1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        # x: [B, N, D]
        A = self.adj_constructor(x)  # [B, N, N]

        # GCN 层 1
        out = torch.matmul(A, x)
        out = self.gcn1(out)
        out = F.relu(out)

        # GCN 层 2
        out = torch.matmul(A, out)
        out = self.gcn2(out)
        out = F.relu(out)

        return out  # [B, N, hidden_dim] [B, N, N]


class AGCLSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, dropout, T_in, T_out):
        super(AGCLSTM, self).__init__()
        self.agcn = AdaptiveGCN(in_channels, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # self.linear1 = nn.Linear(in_features=hidden_size, out_features=256)
        # self.linear2 = nn.Linear(in_features=256, out_features=128)
        # self.linear3 = nn.Linear(in_features=128, out_features=64)
        # self.dropout = nn.Dropout(dropout)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x):
        """
        :param x: [B, N, T_in, D]
        :return: [B, N, T_out]
        """
        B, N, t, d = x.size()
        out = []
        for i in range(t):
            out.append(self.agcn(x[:, :, i, :]))
        out = torch.stack(out, dim=2)
        out = out.reshape(B*N, t, -1)
        out, _ = self.lstm(out)
        out = out.reshape(B, N, t, -1)
        out = out.permute(0, 2, 1, 3)
        # out = F.relu(self.dropout(self.linear1(out)))
        # out = F.relu(self.dropout(self.linear2(out)))
        # out = F.relu(self.dropout(self.linear3(out)))
        out = self.prediction_layer(out)
        return out