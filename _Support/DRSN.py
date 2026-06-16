import torch
import torch.nn as nn


class RSBU_CW(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
        super().__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)

    def forward(self, input):
        x = self.BRC(input)
        x_abs = torch.abs(x)
        gap = self.global_average_pool(x_abs)
        gap = self.flatten(gap)
        alpha = self.FC(gap)
        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)
        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x), n_sub)
        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            zero_padding = torch.zeros(input.shape).cuda()
            input = torch.cat((input, zero_padding), dim=1)

        result = x + input
        return result


class DRSNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear6_8 = nn.Linear(in_features=256, out_features=128)
        self.linear8_4 = nn.Linear(in_features=128, out_features=64)
        self.linear4_2 = nn.Linear(in_features=64, out_features=32)
        self.output_center_pos = nn.Linear(in_features=32, out_features=1)
        self.output_width = nn.Linear(in_features=32, out_features=1)

        self.linear = nn.Linear(in_features=16, out_features=8)
        self.output_class = nn.Linear(in_features=8, out_features=3)

    def forward(self, input):  # 1*256
        x = self.conv1(input.permute(0, 2, 1))  # 4*128
        x = RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=True).cuda()(x)  # 4*64
        x = RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=False).cuda()(x)  # 4*64
        x = RSBU_CW(in_channels=4, out_channels=8, kernel_size=3, down_sample=True).cuda()(x)  # 8*32
        x = RSBU_CW(in_channels=8, out_channels=8, kernel_size=3, down_sample=False).cuda()(x)  # 8*32
        x = RSBU_CW(in_channels=8, out_channels=16, kernel_size=3, down_sample=True).cuda()(x)  # 16*16
        x = RSBU_CW(in_channels=16, out_channels=16, kernel_size=3, down_sample=False).cuda()(x)  # 16*16
        x = self.bn(x)
        x = self.relu(x)
        gap = self.global_average_pool(x)  # 16*1
        gap = self.flatten(gap)  # 1*16
        linear1 = self.linear(gap).unsqueeze(2)  # 1*8
        # output_class = self.output_class(linear1)  # 1*3
        # output_class = self.softmax(output_class)  # 1*3

        # return output_class
        return linear1[:, -1:]
