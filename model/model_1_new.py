import torch
import torch.nn as nn
import torch.nn.functional as F
from _Support.TemporalConvNet import TemporalConvNet
from GNN.GATLayer import GAT
from GNN.GCNLayer import GCNLayer
from model.model import PredictionLayer


class GraphAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_graphs=3):
        super(GraphAttentionFusion, self).__init__()
        self.num_graphs = num_graphs

        # 用于计算图级注意力权重
        self.attn_fc = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, graph_features):
        """
        graph_features: list of tensors
            length = num_graphs
            each tensor shape = [B, N, T, D]
        """

        # 1. 堆叠不同图的特征
        # H: [B, N, T, K, D]
        H = torch.stack(graph_features, dim=3)

        # 2. 计算注意力得分
        # score: [B, N, T, K, 1] -> [B, N, T, K]
        score = self.attn_fc(H).squeeze(-1)

        # 3. Softmax 归一化（在图维度）
        alpha = F.softmax(score, dim=3)

        # 4. 加权求和
        # alpha: [B, N, T, K, 1]
        alpha = alpha.unsqueeze(-1)

        # H_fused: [B, N, T, D]
        H_fused = torch.sum(alpha * H, dim=3)

        return H_fused


class gated_TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_channels = num_channels
        self.TCN1 = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.TCN2 = TemporalConvNet(input_size, num_channels, kernel_size, dropout)

    def forward(self, x):
        """
        :param x:(batch_size, seq_len, num_nodes, features)
        :return: (batch_size, seq_len, num_nodes, num_channels[-1])
        """
        x = x.transpose(1, 2)
        B, N, t, d = x.shape
        x = x.reshape(B * N, t, d)
        x = x.transpose(1, 2)
        TCN1_output = F.tanh(self.TCN1(x))
        TCN2_output = F.sigmoid(self.TCN2(x))
        output = (TCN1_output * TCN2_output).transpose(1, 2)
        output = output.contiguous().view(B, N, t, self.num_channels[-1]).permute(0, 2, 1, 3)
        return output


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_channels = num_channels
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        B, N, t, d = x.shape
        x = x.reshape(B * N, t, d)
        x = x.transpose(1, 2)
        output = self.tcn(x).transpose(1, 2)
        output = output.contiguous().view(B, N, t, self.num_channels[-1]).permute(0, 2, 1, 3)
        return output



class SpitalBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, adj, dropout, alpha, n_heads, gcn_bool, gat_bool):
        super(SpitalBlock, self).__init__()
        self.gcn_bool = gcn_bool
        self.gat_bool = gat_bool
        self.adj = adj
        self.gcn = GCNLayer(in_channels, hidden_size, hidden_size, adj, bias=True)
        self.gat = GAT(in_channels, hidden_size, dropout, alpha, n_heads)

        self.f_gcn = nn.Linear(hidden_size, hidden_size)
        self.f_gat = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        if self.gcn_bool and self.gat_bool:
            out_gcn = self.gcn(x)
            out_gat = self.gat(x, self.adj)
            gate = torch.sigmoid(self.f_gcn(out_gcn) + self.f_gat(out_gat))
            out = gate * out_gcn + (1 - gate) * out_gat
        elif self.gcn_bool:
            out = self.gcn(x)
        elif self.gat_bool:
            out = self.gat(x, self.adj)
        return out


class ComputeAttentionScore(nn.Module):
    def __init__(self):
        super(ComputeAttentionScore, self).__init__()

    def forward(self, x, node_vec):
        n_q = node_vec.unsqueeze(dim=-1)
        x_t_a = torch.einsum('btnd,ndl->btnl', (x, n_q)).contiguous()
        return x_t_a


class spatial_temporal_block(nn.Module):
    def __init__(self, in_channels, hidden_size, num_channels, dropout, alpha, n_heads, kernel_size, num_nodes, apt_size, adj, gated_TCN_bool, gcn_bool, gat_bool, ASTAM_bool):
        super(spatial_temporal_block, self).__init__()
        # 参数
        self.gated_TCN_bool = gated_TCN_bool
        self.ASTAM_bool = ASTAM_bool
        # 扩展维度
        self.start_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=(1, 1))
        # 时间模块
        self.gated_TCN = gated_TCN(hidden_size, num_channels, kernel_size, dropout)
        self.TCN = TCN(hidden_size, num_channels, kernel_size, dropout)
        # 空间模块
        self.spital_block = SpitalBlock(hidden_size, hidden_size, adj, dropout, alpha, n_heads, gcn_bool, gat_bool)
        # 时空异质性建模
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, apt_size).cuda(), requires_grad=True).cuda()
        self.ComputeAttentionScore = ComputeAttentionScore()
        self.w_s = (nn.Conv2d(in_channels=apt_size, out_channels=hidden_size, kernel_size=(1, 1)))
        self.w_t = (nn.Conv2d(in_channels=apt_size, out_channels=hidden_size, kernel_size=(1, 1)))

    def forward(self, x):
        """
        :param x: (batch_size, num_nodes, seq_len, in_dim)
        :return: (batch_size, num_nodes, seq_len. hidden_size)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.start_conv(x).permute(0, 3, 2, 1)
        if self.gated_TCN_bool:
            x_t = self.gated_TCN(x)
        else:
            # x_t = self.TCN(x)
            x_t = x
        x_s = self.spital_block(x_t)
        if self.ASTAM_bool:
            # 计算时间维度的注意力分数，ComputeAttentionScore输入为(B, T, N, F)。输出为(B, T, N, F)
            n_q_t = self.w_t(self.node_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze()  # 时间维度查询矩阵query，通过节点嵌入计算得到 (num_nodes, hidden_size)
            x_t_a = self.ComputeAttentionScore(x_t, n_q_t)  # 时间维度注意力分数(通过时间卷积模块计算得到)

            # 计算空间维度的注意力分数
            n_q_s = self.w_s(self.node_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze()  # 空间维度查询矩阵query (num_nodes, hidden_size)
            x_s_a = self.ComputeAttentionScore(x_s, n_q_s)  # 空间维度注意力分数(通过空间模块计算得到)

            # node-level adaptation tendencies
            x_a = torch.cat((x_t_a, x_s_a), -1)
            x_att = F.softmax(x_a, dim=-1)

            # Add Temporal, Spatial attention
            x = x_att[:, :, :, 0].unsqueeze(dim=-1) * x_t + x_att[:, :, :, 1].unsqueeze(dim=-1) * x_s
            x = x.transpose(1, 2)
        else:
            x = x_t.transpose(1, 2) + x_s.transpose(1, 2)
        return x


class Model(nn.Module):
    def __init__(self, adj_Distance, adj_distributional_similarity, adj_functional_similarity, input_size, hidden_size, dropout, alpha, n_heads, kernel_size, num_channels, apt_size, num_nodes, num_block, predict_len, gated_TCN_bool, gcn_bool, gat_bool, ASTAM_bool):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.predict_len = predict_len

        self.Blocklist_adj_distance = nn.ModuleList([
            spatial_temporal_block(in_channels=input_size, hidden_size=hidden_size, num_channels=num_channels, dropout=dropout,
                           alpha=alpha, n_heads=n_heads, kernel_size=kernel_size, num_nodes=num_nodes,
                           apt_size=apt_size, adj=adj_Distance,
                           gated_TCN_bool=gated_TCN_bool, gcn_bool=gcn_bool, gat_bool=gat_bool, ASTAM_bool=ASTAM_bool)
            if i == 0 else
            spatial_temporal_block(in_channels=hidden_size, hidden_size=hidden_size, num_channels=num_channels, dropout=dropout,
                           alpha=alpha, n_heads=n_heads, kernel_size=kernel_size, num_nodes=num_nodes,
                           apt_size=apt_size, adj=adj_Distance,
                           gated_TCN_bool=gated_TCN_bool, gcn_bool=gcn_bool, gat_bool=gat_bool, ASTAM_bool=ASTAM_bool)
            for i in range(num_block)
        ])

        self.Blocklist_adj_distributional_similarity = nn.ModuleList([
            spatial_temporal_block(in_channels=input_size, hidden_size=hidden_size, num_channels=num_channels, dropout=dropout,
                           alpha=alpha, n_heads=n_heads, kernel_size=kernel_size, num_nodes=num_nodes,
                           apt_size=apt_size, adj=adj_distributional_similarity,
                           gated_TCN_bool=gated_TCN_bool, gcn_bool=gcn_bool, gat_bool=gat_bool, ASTAM_bool=ASTAM_bool)
            if i == 0 else
            spatial_temporal_block(in_channels=hidden_size, hidden_size=hidden_size, num_channels=num_channels, dropout=dropout,
                           alpha=alpha, n_heads=n_heads, kernel_size=kernel_size, num_nodes=num_nodes,
                           apt_size=apt_size, adj=adj_distributional_similarity,
                           gated_TCN_bool=gated_TCN_bool, gcn_bool=gcn_bool, gat_bool=gat_bool, ASTAM_bool=ASTAM_bool)
            for i in range(num_block)
        ])

        self.Blocklist_adj_functional_similarity = nn.ModuleList([
            spatial_temporal_block(in_channels=input_size, hidden_size=hidden_size, num_channels=num_channels, dropout=dropout,
                           alpha=alpha, n_heads=n_heads, kernel_size=kernel_size, num_nodes=num_nodes,
                           apt_size=apt_size, adj=adj_functional_similarity,
                           gated_TCN_bool=gated_TCN_bool, gcn_bool=gcn_bool, gat_bool=gat_bool, ASTAM_bool=ASTAM_bool)
            if i == 0 else
            spatial_temporal_block(in_channels=hidden_size, hidden_size=hidden_size, num_channels=num_channels, dropout=dropout,
                           alpha=alpha, n_heads=n_heads, kernel_size=kernel_size, num_nodes=num_nodes,
                           apt_size=apt_size, adj=adj_functional_similarity,
                           gated_TCN_bool=gated_TCN_bool, gcn_bool=gcn_bool, gat_bool=gat_bool, ASTAM_bool=ASTAM_bool)
            for i in range(num_block)
        ])

        self.graph_attention_fusion = GraphAttentionFusion(hidden_size, num_graphs=3)
        self.prediction_layer = PredictionLayer(T_dim=24, output_T_dim=predict_len, embed_size=hidden_size)

    def forward(self, x):
        """
        x: [B, N, T, D]
        """

        # ========= 1. 三个图分支并行 =========
        x_distance = x
        x_distributional = x
        x_functional = x

        # Distance graph
        for block in self.Blocklist_adj_distance:
            x_distance = block(x_distance)

        # Distributional similarity graph
        for block in self.Blocklist_adj_distributional_similarity:
            x_distributional = block(x_distributional)

        # Functional similarity graph
        for block in self.Blocklist_adj_functional_similarity:
            x_functional = block(x_functional)

        # 此时三个张量形状均为：
        # [B, N, T, hidden_size]

        # ========= 2. 图级注意力融合 =========
        H_fused = self.graph_attention_fusion([
            x_distance,
            x_distributional,
            x_functional
        ])
        # H_fused: [B, N, T, hidden_size]

        # ========= 3. 预测层 =========
        # PredictionLayer 通常期望 [B, T, N, D]
        H_fused = H_fused.permute(0, 2, 1, 3)

        out = self.prediction_layer(H_fused)

        return out
