import torch
import math
import torch.nn as nn
import torch.nn.init as init
from torch.nn import ModuleList
import torch.nn.functional as F

from _Support.Graph_Construction_Delhi import calculate_the_distance_matrix


adj_matrix, _, _ = calculate_the_distance_matrix(threshold=0.8)
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float).cuda()


class gated_TCN(nn.Module):
    def __init__(self, residual_channels=32, dilation_channels=32, kernel_size=2, layers=2):
        super(gated_TCN, self).__init__()
        self.layers = layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        new_dilation = 1
        for _ in range(layers):
            # dilated convolutions
            self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                               out_channels=dilation_channels,
                                               kernel_size=(1, kernel_size), dilation=new_dilation))

            self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                             out_channels=dilation_channels,
                                             kernel_size=(1, kernel_size), dilation=new_dilation))

            new_dilation *= 2

    def forward(self, input):
        x = input
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
        return x


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, adj_matrix):
        """
        :param x: (batch_size, seq_len, num_nodes, in_channels)
        :return: (batch_size, seq_len, num_nodes, out_channels)
        """
        first_mul = torch.einsum('hi,btij->bthj', adj_matrix, x)
        second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul

        return graph_conv


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # attention——> [B, N, N]
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(2, 3)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_in, n_out, dropout, alpha, nheads, order=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.order = order

        self.attentions = [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        for k in range(2, self.order + 1):
            self.attentions_2 = ModuleList(
                [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                 range(nheads)])

        self.out_att = GraphAttentionLayer(n_out * nheads * order, n_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        for k in range(2, self.order + 1):
            x2 = torch.cat([att(x, adj) for att in self.attentions_2], dim=-1)
            x = torch.cat([x, x2], dim=-1)
        x = F.elu(self.out_att(x, adj))
        return x


class Gate(nn.Module):
    def __init__(self, n_out):
        """Dense version of GAT."""
        super(Gate, self).__init__()
        self.n_out = n_out

        self.W_z = nn.Parameter(torch.empty(size=(2 * n_out, n_out)))
        nn.init.xavier_uniform_(self.W_z.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(1, n_out)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, x, h):
        x_h = torch.cat((x, h), dim=-1)  # concat x and h_(t-1)
        Wh = torch.matmul(x_h, self.W_z)
        gate = torch.sigmoid(Wh + self.b)
        one_vec = torch.ones_like(gate)
        z = gate * x + (one_vec - gate) * h
        return z


class ComputeAttentionScore(nn.Module):
    def __init__(self):
        super(ComputeAttentionScore, self).__init__()

    def forward(self, x, node_vec):
        n_q = node_vec.unsqueeze(dim=-1)
        x_t_a = torch.einsum('btnd,ndl->btnl', (x, n_q)).contiguous()
        return x_t_a


class Model_block(nn.Module):
    def __init__(self, in_dim, num_nodes, apt_size, residual_channels, dilation_channels, kernel_size, layers, dropout,
                 alpha, n_heads, gcn_bool=True, gat_bool=True, stam_bool=True):
        super(Model_block, self).__init__()
        self.gcn_bool = gcn_bool
        self.gat_bool = gat_bool
        self.stam_bool = stam_bool
        self.ComputeAttentionScore = ComputeAttentionScore()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.gated_TCN = gated_TCN(residual_channels, dilation_channels, kernel_size, layers)
        self.gcn = GraphConv(in_channels=dilation_channels, out_channels=dilation_channels)
        self.gat = GAT(n_in=dilation_channels, n_out=dilation_channels, dropout=dropout, alpha=alpha, nheads=n_heads)

        self.node_vec1 = nn.Parameter(torch.randn(num_nodes, apt_size).cuda(), requires_grad=True).cuda()
        self.node_vec2 = nn.Parameter(torch.randn(apt_size, num_nodes).cuda(), requires_grad=True).cuda()

        self.w_t = (nn.Conv2d(in_channels=apt_size, out_channels=dilation_channels, kernel_size=(1, 1)))
        self.w_s = (nn.Conv2d(in_channels=apt_size, out_channels=dilation_channels, kernel_size=(1, 1)))

        self.gate = Gate(residual_channels)

        self.residual_conv = nn.Conv2d(in_channels=dilation_channels, out_channels=dilation_channels, kernel_size=(1, 1))

    def forward(self, x):
        """
        :param x: (batch_size, in_dim, num_nodes, seq_len)
        :return: (batch_size, dilation_channels, num_nodes, seq_len)
        """
        # 时间卷积模块
        x = self.start_conv(x)   # (batch_size, residual_channels, num_nodes, seq_len)
        x_t = self.gated_TCN(x)   # (batch_size, dilation_channels, num_nodes, seq_len)
        x_t = x_t.transpose(1, 3)   # (batch_size, seq_len, num_nodes, dilation_channels)

        # 计算时间维度的注意力分数，ComputeAttentionScore输入为(B, T, N, F)。输出为(B, T, N, F)
        n_q_t = self.w_t(self.node_vec1.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze()  # 时间维度查询矩阵query，通过节点嵌入计算得到 (num_nodes, dilation_channels)
        x_t_a = self.ComputeAttentionScore(x_t, n_q_t)  # 时间维度注意力分数(通过时间卷积模块计算得到)

        if self.gcn_bool:
            # 图卷积模块，图卷积输入为(B, T, N, F)，输出为(B, T, N, F)
            adp = F.softmax(F.relu(torch.mm(self.node_vec1, self.node_vec2)), dim=1)
            adp = torch.eye(adp.shape[0]).to(adp.device) + adp
            x_gcn = self.gcn(x_t, adp)   # (batch_size, seq_len, num_nodes, dilation_channels)
        else:
            x_gcn = self.residual_conv(x_t.transpose(1, 3)).transpose(1, 3)   # (batch_size, seq_len, num_nodes, dilation_channels)

        if self.gat_bool:
            # 图注意力模块，图注意力输入为(B, T, N, F)，输出为(B, T, N, F)
            x_gat = self.gat(x_t, adj_matrix)   # (batch_size, seq_len, num_nodes, dilation_channels)
        else:
            x_gat = self.residual_conv(x_t.transpose(1, 3)).transpose(1, 3)   # (batch_size, seq_len, num_nodes, dilation_channels)

        # 空间模块门控机制，输入为(B, T, N, F)，输出为(B, T, N, F)
        x_s = self.gate(x_gcn, x_gat)   # (batch_size, seq_len, num_nodes, dilation_channels)

        # 计算空间维度的注意力分数
        n_q_s = self.w_s(self.node_vec1.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze()  # 空间维度查询矩阵query (num_nodes, dilation_channels)
        x_s_a = self.ComputeAttentionScore(x_s, n_q_s)  # 空间维度注意力分数(通过空间模块计算得到)

        # node-level adaptation tendencies
        x_a = torch.cat((x_t_a, x_s_a), -1)
        x_att = F.softmax(x_a, dim=-1)

        if self.stam_bool:
            # Add Temporal, Spatial attention
            x = x_att[:, :, :, 0].unsqueeze(dim=-1) * x_t + x_att[:, :, :, 1].unsqueeze(dim=-1) * x_s
            x = x.transpose(1, 3)  # (batch_size, dilation_channels, num_nodes, seq_len)
        else:
            x = (x_t + x_s).transpose(1, 3)   # (batch_size, dilation_channels, num_nodes, seq_len)

        return x   # (batch_size, dilation_channels, num_nodes, seq_len)
    
    
class Model(nn.Module):
    def __init__(self, in_dim, num_nodes, apt_size, residual_channels, dilation_channels, kernel_size, layers, dropout,
                 alpha, n_heads, block_num, end_channels, predict_len, gcn_bool, gat_bool, stam_bool):
        super(Model, self).__init__()
        self.predict_len = predict_len
        self.recent_model = nn.ModuleList([Model_block(in_dim, num_nodes, apt_size, residual_channels, dilation_channels, kernel_size, layers, dropout, alpha, n_heads, gcn_bool, gat_bool, stam_bool)
                                           if i == 0 else
                                           Model_block(dilation_channels, num_nodes, apt_size, residual_channels, dilation_channels, kernel_size, layers, dropout, alpha, n_heads, gcn_bool, gat_bool, stam_bool)
                                           for i in range(block_num)
                                           ])
        self.day_model = nn.ModuleList([Model_block(in_dim, num_nodes, apt_size, residual_channels, dilation_channels, kernel_size, layers, dropout, alpha, n_heads, gcn_bool, gat_bool, stam_bool)
                                           if i == 0 else
                                           Model_block(dilation_channels, num_nodes, apt_size, residual_channels, dilation_channels, kernel_size, layers, dropout, alpha, n_heads, gcn_bool, gat_bool, stam_bool)
                                           for i in range(block_num)
                                           ])
        self.week_model = nn.ModuleList([Model_block(in_dim, num_nodes, apt_size, residual_channels, dilation_channels, kernel_size, layers, dropout, alpha, n_heads, gcn_bool, gat_bool, stam_bool)
                                           if i == 0 else
                                           Model_block(dilation_channels, num_nodes, apt_size, residual_channels, dilation_channels, kernel_size, layers, dropout, alpha, n_heads, gcn_bool, gat_bool, stam_bool)
                                           for i in range(block_num)
                                           ])

        self.mlp = nn.Sequential(
            # nn.Conv2d(in_channels=dilation_channels * 3, out_channels=end_channels, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=dilation_channels, out_channels=end_channels, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=end_channels, out_channels=1, kernel_size=(1, 1))
        )

    def forward(self, x1, x2, x3):
        for i in range(len(self.recent_model)):
            x1 = self.recent_model[i](x1)

        # for i in range(len(self.day_model)):
        #     x2 = self.day_model[i](x2)
        #
        # for i in range(len(self.week_model)):
        #     x3 = self.week_model[i](x3)

        # x = torch.cat((x1, x2, x3), dim=1)
        x = self.mlp(x1)   # (batch_size, 1, num_nodes, seq_len)
        return x[:, :, :, -self.predict_len:].permute(0, 2, 3, 1)   # (B, N, T, F)