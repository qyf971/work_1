import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import scipy.sparse as sp


# 将邻接矩阵转为PyG要求格式
def convert_adj_sparse(adj):
    adj_coo = sp.coo_matrix(adj.cpu().numpy())
    edge_attr = torch.FloatTensor(adj_coo.data).to(adj.device)
    edge_index = torch.LongTensor([adj_coo.row, adj_coo.col]).to(adj.device)
    return edge_index, edge_attr


class GAT(nn.Module):
    def __init__(self, device, in_features, out_features, edge_dim=1):
        super(GAT, self).__init__()
        self.gat = GATConv(in_features, out_features, edge_dim=edge_dim).to(device)
        self.convert = convert_adj_sparse

    def forward(self, data, adj):
        edge_index, edge_attr = self.convert(adj)
        batch_size = data.size(0)
        num_nodes = adj.size(0)

        x = data.reshape(batch_size * num_nodes, -1)

        # 重复 edge_index
        edge_index = edge_index.repeat(1, batch_size)

        # 重复 edge_attr
        edge_attr = edge_attr.repeat(batch_size)

        x, attention_weights = self.gat(x, edge_index, edge_attr, return_attention_weights=True)

        x = x.reshape(batch_size, num_nodes, -1)
        return x, attention_weights


# class GAT_Layer(nn.Module):
#     def __init__(self, device, in_features, out_features, edge_dim=1):
#         super(GAT_Layer, self).__init__()
#         self.gat = GAT(device, in_features, out_features, edge_dim=edge_dim)
#
#     def forward(self, data, adj):
#         batch_size, T, num_nodes, in_features = data.shape
#         outputs = []
#
#         for t in range(T):
#             gat_output, _ = self.gat(data[:, t], adj)
#             outputs.append(gat_output)
#
#         # 将所有时间步的输出堆叠起来
#         outputs = torch.stack(outputs, dim=1)
#         return outputs


class GAT_Layer(nn.Module):
    def __init__(self, device, in_features, out_features, edge_dim=1):
        super(GAT_Layer, self).__init__()
        self.gat = GAT(device, in_features, out_features, edge_dim=edge_dim)

    def forward(self, data, adj):
        B, T, N, d = data.shape

        # 将数据展平为 (B * T, N, d)
        data_flattened = data.reshape(B * T, N, d)

        # 一次性传入 GAT 层
        x, _ = self.gat(data_flattened, adj)

        # 将结果恢复为 (B, T, N, out_features)
        x = x.reshape(B, T, N, -1)

        return x



