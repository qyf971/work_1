import torch.nn as nn

from _Support.CBAM import CBAM
from _Support.TemporalConvNet import TemporalConvNet


class STARes(nn.Module):
    def __init__(self, in_channels, ratio):
        super(STARes, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.CBAM = CBAM(in_channels, ratio)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.CBAM(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        return output + x


class STCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(STCN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=64, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout=dropout)

    def forward(self, x):
        output = self.conv(x)
        output = output.squeeze(1)
        output = self.tcn(output.transpose(1, 2)).transpose(1, 2)
        return output


class STARes_STCN(nn.Module):
    def __init__(self, in_channels, input_len, ratio, num_channels, kernel_size, dropout, predict_len,
                 STARes_blocks):
        super(STARes_STCN, self).__init__()
        self.predict_len = predict_len
        self.sta_res_BlockList = nn.ModuleList([
            STARes(input_len, ratio)
            for _ in range(STARes_blocks)
        ])
        self.STCN = STCN(in_channels, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        for i in range(len(self.sta_res_BlockList)):
            x = self.sta_res_BlockList[i](x)
        x = self.STCN(x.permute(0, 2, 1, 3))
        x = self.linear(x)
        x = x[:, -self.predict_len:, :]
        return x
