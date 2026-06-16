import torch
import torch.nn as nn
from model.model import PredictionLayer
from _Support.CBAM import ChannelAttention, SpatialAttention, CBAM


class CBAM_CNN_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, T_in, T_out):
        super(CBAM_CNN_BiLSTM, self).__init__()
        # Attention
        self.cbam = CBAM(input_size)

        # CNN layers
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Prediction layer
        self.prediction_layer = PredictionLayer(
            T_dim=T_in,
            output_T_dim=T_out,
            embed_size=hidden_size * 2   # 因为是双向LSTM
        )

    def forward(self, x):
        """
        x: [B, N, T, D]
        """
        B, N, T, D = x.shape

        # 👉 转为 CNN 输入格式
        x = x.permute(0, 3, 1, 2)   # [B, D, N, T]

        # 👉 CBAM 注意力
        x = self.cbam(x)            # [B, D, N, T]

        # 👉 CNN 提取空间特征
        x = self.relu(self.conv1(x))   # [B, 32, N, T]
        x = self.relu(self.conv2(x))   # [B, 64, N, T]

        # 👉 调整为 LSTM 输入
        x = x.permute(0, 2, 3, 1)      # [B, N, T, 64]

        # ⭐ 展开节点维度（关键步骤）
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C)     # [B*N, T, 64]

        # 👉 BiLSTM 建模时间依赖
        x, _ = self.bilstm(x)          # [B*N, T, 2*hidden]

        # 👉 恢复节点结构
        x = x.reshape(B, N, T, -1)     # [B, N, T, 2*hidden]

        x = x.transpose(1, 2)              # [B, N, 2*hidden, T]

        # 👉 预测层
        x = self.prediction_layer(x)   # [B, T_out, N, D]（取决于你的实现）

        return x