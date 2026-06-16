import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class AdaptiveGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_nodes, device):
        super(AdaptiveGCN, self).__init__()
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.linear = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        """
        :param x: [B, N, F_in]
        :return: [B, N, F_out]
        """
        adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        x = torch.matmul(adj, x)
        x = self.linear(x)
        x = F.relu(x)
        return x, adj

class PredictionLayer(nn.Module):
    def __init__(self, T_in, T_out, hidden_dim):
        super(PredictionLayer, self).__init__()

        # 缩小时间维度。
        self.conv1 = nn.Conv2d(T_in, T_out, 1)
        # 缩小通道数，降到1维。
        self.conv2 = nn.Conv2d(hidden_dim, 1, 1)
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
    
class ST_Block(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, num_layers, dropout, device):
        super(ST_Block, self).__init__()
        self.agcn = AdaptiveGCN(input_dim, hidden_dim, num_nodes, device)

        # 可替换的时序组件
        # self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.GRU = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout)

    def forward(self, x):
        B, N, T, _ = x.shape

        x_out = []
        for t in range(T):
            x_t = x[:, :, t, :]  # [B, N, F]
            x_t_out, _ = self.agcn(x_t)
            x_out.append(x_t_out)
        x_out = torch.stack(x_out, dim=2)

        x_out = x_out.view(B*N, T, -1)
        x_out, _ = self.GRU(x_out)
        x_out = x_out.view(B, N, T, -1)

        # residual connection
        x_out = x_out + x
        return x_out


class Proposed(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, T_in, T_out, num_layers, num_blocks, dropout, device):
        super(Proposed, self).__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)

        self.GRU_recent = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.GRU_day = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.GRU_week = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        self.agcn = AdaptiveGCN(hidden_dim, hidden_dim, num_nodes, device)

        self.mamba = self.mamba = Mamba(
            d_model=hidden_dim,  # Model dimension d_model
            d_state=hidden_dim,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=1,  # Block expansion factor)
        )

        # self.ST_Blocks_recent = nn.ModuleList([
        #     ST_Block(input_dim=hidden_dim, hidden_dim=hidden_dim, num_nodes=num_nodes, num_layers=num_layers, dropout=dropout, device=device)
        #     for _ in range(num_blocks)]
        # )

        # self.ST_Blocks_day = nn.ModuleList([
        #     ST_Block(input_dim=hidden_dim, hidden_dim=hidden_dim, num_nodes=num_nodes, num_layers=num_layers, dropout=dropout, device=device)
        #     for _ in range(num_blocks)]
        # )
        #
        # self.ST_Blocks_week = nn.ModuleList([
        #     ST_Block(input_dim=hidden_dim, hidden_dim=hidden_dim, num_nodes=num_nodes, num_layers=num_layers, dropout=dropout, device=device)
        #     for _ in range(num_blocks)]
        # )

        self.prediction_layer = PredictionLayer(T_in=T_in, T_out=T_out, hidden_dim=hidden_dim)

    def forward(self, x_recent, x_day, x_week):
    # def forward(self, x_recent):
        """
        :param x: [B, N, T_in, F]
        :return: [B, N, T_out]
        """
        x_recent = self.input_embedding(x_recent.permute(0, 2, 3, 1))
        x_day = self.input_embedding(x_day.permute(0, 2, 3, 1))
        x_week = self.input_embedding(x_week.permute(0, 2, 3, 1))

        # for i, block in enumerate(self.ST_Blocks_recent):
        #     x_recent = block(x_recent)

        # for i, block in enumerate(self.ST_Blocks_day):
        #     x_day = block(x_day)
        #
        # for i, block in enumerate(self.ST_Blocks_week):
        #     x_week = block(x_week)

        # x_out = (x_recent + x_day + x_week).transpose(1, 2)

        B, N, T_recent, _ = x_recent.shape

        x_recent = x_recent.view(B*N, T_recent, -1)
        x_day = x_day.view(B*N, T_recent, -1)
        x_week = x_week.view(B*N, T_recent, -1)

        x_recent, _ = self.GRU_recent(x_recent)
        x_day, _ = self.GRU_day(x_day)
        x_week, _ = self.GRU_week(x_week)

        x = x_recent.view(B, N, T_recent, -1) + x_day.view(B, N, T_recent, -1) + x_week.view(B, N, T_recent, -1)

        x_out = []
        for t in range(T_recent):
            x_t = x[:, :, t, :]  # [B, N, F]
            x_t_out, _ = self.agcn(x_t)
            x_out.append(x_t_out)
        x_out = torch.stack(x_out, dim=2)

        x_out = self.mamba(x_out.view(B*N, T_recent, -1)).view(B, N, T_recent, -1)

        x_out = x_out.transpose(1, 2)

        # x_out = x_recent.transpose(1, 2)
        x_out = self.prediction_layer(x_out)
        return x_out













