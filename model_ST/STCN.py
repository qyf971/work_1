import torch.nn as nn
from _Support.TemporalConvNet import TemporalConvNet


class STCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, predict_len):
        super(STCN, self).__init__()
        self.predict_len = predict_len
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=64, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        output = self.conv(x)
        output = output.squeeze(1)
        output = self.tcn(output.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output)
        output = output[:, -self.predict_len:, :]
        return output