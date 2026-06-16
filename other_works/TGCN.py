import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv

class ChebGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, K, num_nodes):
        """
        定义 ChebConv 嵌入 GRU 的单元
        Args:
            in_channels (int): 输入特征维度
            hidden_channels (int): 隐藏状态的特征维度
            K (int): 切比雪夫多项式阶数
            num_nodes (int): 节点数
        """
        super(ChebGRUCell, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes

        # 使用 ChebConv 替代 GRU 中的线性变换
        self.conv_z = ChebConv(in_channels + hidden_channels, hidden_channels, K)  # 更新门
        self.conv_r = ChebConv(in_channels + hidden_channels, hidden_channels, K)  # 重置门
        self.conv_h = ChebConv(in_channels + hidden_channels, hidden_channels, K)  # 候选状态

    def forward(self, x, h, edge_index, edge_weight):
        """
        单步前向传播
        Args:
            x (Tensor): 输入特征 [batch_size, num_nodes, in_channels]
            h (Tensor): 隐藏状态 [batch_size, num_nodes, hidden_channels]
            edge_index (Tensor): 图的边索引
            edge_weight (Tensor): 图的边权重
        Returns:
            h_next (Tensor): 下一时间步的隐藏状态 [batch_size, num_nodes, hidden_channels]
        """
        # 拼接输入和隐藏状态
        combined = torch.cat([x, h], dim=-1)  # [batch_size, num_nodes, in_channels + hidden_channels]

        # 更新门 z_t
        z = torch.sigmoid(self.conv_z(combined, edge_index, edge_weight))  # [batch_size, num_nodes, hidden_channels]

        # 重置门 r_t
        r = torch.sigmoid(self.conv_r(combined, edge_index, edge_weight))  # [batch_size, num_nodes, hidden_channels]

        # 候选隐藏状态 h_tilde
        combined_reset = torch.cat([x, r * h], dim=-1)  # [batch_size, num_nodes, in_channels + hidden_channels]
        h_tilde = torch.tanh(self.conv_h(combined_reset, edge_index, edge_weight))  # [batch_size, num_nodes, hidden_channels]

        # 最终隐藏状态 h_next
        h_next = (1 - z) * h + z * h_tilde  # [batch_size, num_nodes, hidden_channels]

        return h_next


class ChebGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, K, num_nodes, num_layers=1):
        """
        定义多层 ChebGRU 模型
        Args:
            in_channels (int): 输入特征维度
            hidden_channels (int): 隐藏状态的特征维度
            K (int): 切比雪夫多项式阶数
            num_nodes (int): 节点数
            num_layers (int): ChebGRU 层数
        """
        super(ChebGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # 定义每一层的 ChebGRU 单元
        self.layers = nn.ModuleList([
            ChebGRUCell(
                in_channels=in_channels if i == 0 else hidden_channels,
                hidden_channels=hidden_channels,
                K=K,
                num_nodes=num_nodes
            )
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_weight, h_0=None):
        """
        前向传播
        Args:
            x (Tensor): 输入特征 [batch_size, time_steps, num_nodes, in_channels]
            edge_index (Tensor): 图的边索引
            edge_weight (Tensor): 图的边权重
            h_0 (Tensor): 初始隐藏状态 [num_layers, batch_size, num_nodes, hidden_channels]
        Returns:
            h (Tensor): 所有时间步的隐藏状态 [batch_size, time_steps, num_nodes, hidden_channels]
            h_n (Tensor): 最后时间步的隐藏状态 [num_layers, batch_size, num_nodes, hidden_channels]
        """
        batch_size, time_steps, num_nodes, in_channels = x.size()

        if h_0 is None:
            h_0 = torch.zeros((self.num_layers, batch_size, num_nodes, self.hidden_channels), device=x.device)

        h = h_0
        h_out = []

        # 遍历时间步
        for t in range(time_steps):
            x_t = x[:, t, :, :]  # 当前时间步的输入 [batch_size, num_nodes, in_channels]
            h_next = []

            # 遍历每一层的 ChebGRU
            for layer_idx, layer in enumerate(self.layers):
                h_t = h[layer_idx]  # 当前层的隐藏状态
                h_t_next = layer(x_t, h_t, edge_index, edge_weight)  # 更新隐藏状态
                h_next.append(h_t_next)
                x_t = h_t_next  # 当前层的输出作为下一层的输入

            h = torch.stack(h_next)  # 更新所有层的隐藏状态
            h_out.append(h[-1])  # 仅保留最后一层的输出

        h_out = torch.stack(h_out, dim=1)  # 拼接时间步 [batch_size, time_steps, num_nodes, hidden_channels]
        return h_out, h
