import torch
import torch.nn as nn
import torch.nn.functional as F

from _Support.CBAM import CBAM
from _Support.TemporalConvNet import TemporalConvNet


class GCN_TCN_CBAM_block(nn.Module):
    def __init__(self, input_dim, hidden_size, input_len, K, num_channels, kernel_size, dropout, edge_index, edge_weight):
        super(GCN_TCN_CBAM_block, self).__init__()
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.STA = CBAM(input_len)
        self.gcn = ChebConv(in_channels=input_dim, out_channels=hidden_size, K=K)
        self.tcn = TemporalConvNet(hidden_size, num_channels, kernel_size, dropout=dropout)
        self.residual_conv = nn.Conv2d(in_channels=input_dim, out_channels=num_channels[-1], kernel_size=(1, 1),
                                       stride=(1, 1))

    def forward(self, input):
        x = input.transpose(1, 2)
        x = self.STA(x)
        batch_size, seq_len, num_nodes, features = x.size()
        x = x.contiguous().view(batch_size * seq_len, num_nodes, features)
        x = self.gcn(x, edge_index=self.edge_index, edge_weight=self.edge_weight)
        x = x.contiguous().view(batch_size, seq_len, num_nodes, self.hidden_size)
        x = x.transpose(1, 2)
        x = x.transpose(2, 3)
        x = x.contiguous().view(batch_size * num_nodes, self.hidden_size, seq_len)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, num_nodes, seq_len, self.num_channels[-1])
        x = x.permute(0, 3, 1, 2)
        x_residual = self.residual_conv(input.permute(0, 3, 1, 2))
        x = (x + x_residual).permute(0, 2, 3, 1)
        return x


class GCN_TCN_CBAM(nn.Module):
    def __init__(self, input_dim, hidden_size, input_len, K, num_channels, kernel_size, dropout, predict_len,
                 block_num, edge_index, edge_weight):
        super(GCN_TCN_CBAM, self).__init__()
        self.num_channels = num_channels
        self.predict_len = predict_len
        self.Blocklist = nn.ModuleList(
            [GCN_TCN_CBAM_block(input_dim, hidden_size, input_len, K, num_channels, kernel_size, dropout, edge_index, edge_weight)
             if i == 0 else
             GCN_TCN_CBAM_block(num_channels[-1], hidden_size, input_len, K, num_channels, kernel_size, dropout, edge_index, edge_weight)
             for i in range(block_num)])
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        batch_size, num_nodes, seq_len, _ = x.size()
        for i in range(len(self.Blocklist)):
            x = self.Blocklist[i](x)
        x = x.contiguous().view(batch_size * num_nodes, seq_len, self.num_channels[-1])
        x = self.linear(x)
        x = x.contiguous().view(batch_size, num_nodes, seq_len, 1)
        return x[:, :, -self.predict_len:, :]