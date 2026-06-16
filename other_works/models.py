import torch
import torch.nn as nn
from other_works import layers


def normalize_adjacency_matrix(adj_matrix, scale=True):
    """
    对邻接矩阵进行对称归一化和缩放归一化。

    :param adj_matrix: torch.Tensor，邻接矩阵，形状为 [N, N]
    :param scale: bool，是否进行缩放归一化
    :return: torch.Tensor，处理后的拉普拉斯矩阵，形状为 [N, N]
    """
    # 将邻接矩阵转换为浮点类型
    adj_matrix = adj_matrix.float()

    # 计算度矩阵 D
    degree_matrix = torch.diag(adj_matrix.sum(dim=1))

    # 计算 D^(-1/2)
    degree_matrix_inv_sqrt = torch.diag(torch.pow(degree_matrix.diagonal(), -0.5))

    # 处理度为 0 的情况（防止 NaN）
    degree_matrix_inv_sqrt[torch.isnan(degree_matrix_inv_sqrt)] = 0

    # 对称归一化拉普拉斯矩阵
    normalized_laplacian = torch.eye(adj_matrix.size(0)).to(adj_matrix.device) - torch.mm(
        torch.mm(degree_matrix_inv_sqrt, adj_matrix),
        degree_matrix_inv_sqrt
    )

    if scale:
        # 计算特征值的最大值
        lambda_max = torch.linalg.eigvalsh(normalized_laplacian).max().item()
        # 缩放归一化
        normalized_laplacian = 2 * normalized_laplacian / lambda_max - torch.eye(adj_matrix.size(0))

    return normalized_laplacian.to(adj_matrix.device)


class STGCNChebGraphConv(nn.Module):
    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, blocks, n_vertex, Kt, Ks, act_func, graph_conv_type, gso, enable_bias, droprate, n_his):
        super(STGCNChebGraphConv, self).__init__()
        # self.gso = normalize_adjacency_matrix(gso, scale=True)
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(Kt, Ks, n_vertex, blocks[l][-1], blocks[l+1], act_func, graph_conv_type, gso, enable_bias, droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = n_his - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, act_func, enable_bias, droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        """
        :param x: [b, d, t, n]
        :return:
        """
        x = x.permute(0, 3, 2, 1)
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        print(x.shape)
        return x

class STGCNGraphConv(nn.Module):
    # STGCNGraphConv contains 'TGTND TGTND TNFF' structure
    # GraphConv is the graph convolution from GCN.
    # GraphConv is not the first-order ChebConv, because the renormalization trick is adopted.
    # Be careful about over-smoothing.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, blocks, n_vertex, Kt, Ks, act_func, graph_conv_type, gso, enable_bias, droprate, n_his):
        super(STGCNGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(Kt, Ks, n_vertex, blocks[l][-1], blocks[l+1], act_func, graph_conv_type, gso, enable_bias, droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = n_his - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, act_func, enable_bias, droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=enable_bias)
            self.relu = nn.ReLU()
            self.do = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        return x
