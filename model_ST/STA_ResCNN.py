import torch.nn as nn
import torch
from _Support.CBAM import ChannelAttention, SpatialAttention


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        if self.downsample is not None:
            residual = self.downsample(residual)
        output += residual
        output = self.relu(output)
        return output


class STA_ResCNN(nn.Module):
    def __init__(self, hidden_size, input_len, ratio, predict_len, resnet_block):
        super(STA_ResCNN, self).__init__()
        self.predict_len = predict_len
        self.channel_attention = ChannelAttention(input_len, ratio)
        self.spatial_attention = SpatialAttention()
        self.Res_Net = nn.ModuleList([
            ResidualBlock(input_len, hidden_size) if i == 0 else
            ResidualBlock(hidden_size, hidden_size) if i < resnet_block - 1 else
            ResidualBlock(hidden_size, 1)
            for i in range(resnet_block)
        ])
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        channel_attention = self.channel_attention(x)
        spatial_attention = self.spatial_attention(x)
        channel_spatial_attention = torch.mul(channel_attention, spatial_attention)
        output = torch.mul(x, channel_spatial_attention)
        for i in range(len(self.Res_Net)):
            output = self.Res_Net[i](output)
        output = self.linear(output.squeeze(1))

        return output[:, -self.predict_len:, :]
