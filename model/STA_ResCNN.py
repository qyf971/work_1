import torch
import torch.nn as nn
from _Support.CBAM import ChannelAttention, SpatialAttention

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

# ================================
# 你提供的 论文原版 ResBlock
# ================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        residual = x
        out = self.tanh(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.tanh(out)
        return out

# ================================
# 你提供的 论文原版 ResCNN
# ================================
class ResCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(ResCNN, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        self.res_block1 = ResBlock(out_channels)
        self.res_block2 = ResBlock(out_channels)

    def forward(self, x):
        x = self.tanh(self.conv_in(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x

# ================================
# 集成版：STA + 论文原版 ResCNN
# ================================
class STA_ResCNN(nn.Module):
    def __init__(self, in_channels, hidden_size, input_len, ratio, predict_len):
        super(STA_ResCNN, self).__init__()
        self.predict_len = predict_len
        
        # 注意力模块
        self.channel_attention = ChannelAttention(input_len, ratio)
        self.spatial_attention = SpatialAttention()
        
        # ✅ 直接使用你要的论文 ResCNN
        self.rescnn = ResCNN(in_channels=in_channels, out_channels=hidden_size)
        
        # 输出层
        self.prediction_layer = PredictionLayer(T_dim=input_len, output_T_dim=predict_len, embed_size=hidden_size)
    def forward(self, x):
        # 维度转换 [B, N, T, F] -> [B, T, N, F]
        x = x.transpose(1, 2)  
        
        # 注意力
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        x = torch.mul(x, torch.mul(ca, sa))

        x = x.permute(0, 3, 1, 2) # [B, F, T, N]
        
        # ✅ 核心：这里直接用论文 ResCNN 提取特征
        x = self.rescnn(x) # # [B, F, T, N]

        x = x.permute(0, 2, 3, 1)

        x = self.prediction_layer(x)
        
        # 输出预测
        return x