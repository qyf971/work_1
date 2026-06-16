import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN.GNN import ChebConv_Layer
from _Support.TemporalConvNet import TemporalConvNet


class PredictionLayer(nn.Module):
    def __init__(self, T_dim, output_T_dim, embed_size):
        super(PredictionLayer, self).__init__()

        # 缩小时间维度。
        self.conv1 = nn.Conv2d(T_dim, output_T_dim, 1)
        # 缩小通道数，降到1维。
        self.conv2 = nn.Conv2d(embed_size, 1, 1)

    def forward(self, input_prediction_layer):
        """
        :param input_prediction_layer: [B, T, N, D]
        :return: [B, N, out_T]
        """
        out = self.conv1(input_prediction_layer) # 等号左边 out shape: [B, T, N, d]
        out = out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, d, N, T]
        out = self.conv2(out)  # 等号左边 out shape: [B, 1, N, T]
        out = out.squeeze(1)

        return out

class GCN_TCN_block(nn.Module):
    def __init__(self, device, adj, input_dim, hidden_size, K, num_channels, kernel_size, dropout):
        super(GCN_TCN_block, self).__init__()
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.gcn = ChebConv_Layer(device, adj, input_dim, hidden_size, K)
        self.tcn = TemporalConvNet(hidden_size, num_channels, kernel_size, dropout=dropout)
        self.residual_conv = nn.Conv2d(in_channels=input_dim, out_channels=num_channels[-1], kernel_size=(1, 1),
                                       stride=(1, 1))

    def forward(self, input):
        x = input.transpose(1, 2)
        batch_size, seq_len, num_nodes, features = x.size()
        x = x.contiguous().view(batch_size * seq_len, num_nodes, features)
        x = self.gcn(x)
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


class STCN(nn.Module):
    def __init__(self, device, adj, input_size, hidden_size, K, num_channels, kernel_size, dropout, predict_len):
        super(STCN, self).__init__()
        self.num_channels = num_channels
        self.predict_len = predict_len
        
        # 固定 3 个 block
        self.block1 = GCN_TCN_block(device, adj, input_size, hidden_size, K, num_channels, kernel_size, dropout)
        self.block2 = GCN_TCN_block(device, adj, num_channels[-1], hidden_size, K, num_channels, kernel_size, dropout)
        self.block3 = GCN_TCN_block(device, adj, num_channels[-1], hidden_size, K, num_channels, kernel_size, dropout)

        self.prediction_layer = PredictionLayer(72, predict_len, num_channels[-1])

    def forward(self, x):
        batch_size, num_nodes, seq_len, _ = x.size()
        
        # 三个块顺序传播
        out1 = self.block1(x)        # 第一个块输出（用于残差）
        out2 = self.block2(out1)     # 第二个块
        out3 = self.block3(out2)     # 第三个块输出
        
        # ===================== 残差连接：block1 + block3 =====================
        residual = out1 + out3
        
        # 送入预测层
        residual = residual.permute(0, 2, 1, 3)
        out = self.prediction_layer(residual)
        
        return out