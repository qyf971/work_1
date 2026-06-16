import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from GNN.GAT import GAT_Layer


class Temporal_Conv_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, recent_only=False):
        super(Temporal_Conv_Layer, self).__init__()
        self.recent_only = recent_only
        self.temporal_conv_layer_recent = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.temporal_conv_layer_day = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.temporal_conv_layer_week = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.mlp = nn.Conv2d(in_channels=out_channels * 3, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, x_recent, x_day, x_week):
        """
        :param x_recent: [B, in_channels, N, T]
        :param x_day: [B, in_channels, N, T]
        :param x_week: [B, in_channels, N, T]
        :return: [B, out_channels, N,, T]
        """
        if self.recent_only:
            out = self.temporal_conv_layer_recent(x_recent)
        else:
            out_recent = self.temporal_conv_layer_recent(x_recent)
            out_day = self.temporal_conv_layer_day(x_day)
            out_week = self.temporal_conv_layer_week(x_week)
            out = torch.cat((out_recent, out_day, out_week), dim=1)
            out = self.mlp(out)
        return out


class TemporalPositionalEncoding(nn.Module):
    """
    Temporal Positional Encoding for the time dimension (T).
    """
    def __init__(self, embed_dim, max_time_len=5000):
        super(TemporalPositionalEncoding, self).__init__()
        # Time positional encoding: [max_time_len, embed_dim]
        pe = torch.zeros(max_time_len, embed_dim)
        position = torch.arange(0, max_time_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('time_pe', pe.unsqueeze(0))  # Add batch dimension

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, N, F].
        Returns:
            Tensor with temporal positional encoding applied.
        """
        B, T, N, F = x.shape
        # Add temporal positional encoding
        time_encoded = self.time_pe[:, :T, :].unsqueeze(2)  # Shape: [1, T, 1, F]
        return x + time_encoded


class TemporalMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TemporalMultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: Tensor of shape [B, T, N, D]
        """
        B, T, N, D = x.shape

        Q = self.q_linear(x)  # [B, T, N, D]
        K = self.k_linear(x)  # [B, T, N, D]
        V = self.v_linear(x)  # [B, T, N, D]

        # 多头分割
        Q = Q.view(B, T, N, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)  # [B, h, N, T, d_k]
        K = K.view(B, T, N, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)  # [B, h, N, T, d_k]
        V = V.view(B, T, N, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)  # [B, h, N, T, d_k]

        # 计算注意力分数
        attn_weights = F.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1
        )  # [B, h, N, T, T]

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)  # [B, h, N, T, d_k]

        # 合并多头输出
        attn_output = attn_output.permute(0, 3, 2, 1, 4).reshape(B, T, N, D)  # [B, T, N, D]

        # 投影回原始维度
        output = self.out_proj(attn_output)  # [B, T, N, D]
        return output


class TemporalTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super(TemporalTransformerEncoderLayer, self).__init__()
        self.self_attn = TemporalMultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Tensor of shape [B, T, N, D]
        """
        # Multi-head Self-Attention
        attn_output = self.self_attn(x)  # [B, T, N, D]
        out1 = self.norm1(x + attn_output)  # Residual + Norm
        out1 = self.dropout1(out1)  # Dropout after LayerNorm

        # Feed-Forward Network
        ffn_output = self.ffn(out1)  # [B, T, N, D]
        out2 = self.norm2(out1 + ffn_output)  # Residual + Norm
        out2 = self.dropout2(out2)  # Dropout after LayerNorm

        return out2


class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, num_layers, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        self.temporal_pe = TemporalPositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            TemporalTransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: Tensor of shape [B, T, N, D]
        """
        x = self.temporal_pe(x)
        for layer in self.layers:
            x = layer(x)
        return x

class SpatialPositionalEncoding(nn.Module):
    """
    Spatial Positional Encoding for the spatial dimension (N).
    """
    def __init__(self, embed_dim, max_space_len=100):
        super(SpatialPositionalEncoding, self).__init__()
        # Space positional encoding: [max_space_len, embed_dim]
        pe = torch.zeros(max_space_len, embed_dim)
        position = torch.arange(0, max_space_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('space_pe', pe.unsqueeze(0))  # Add batch dimension

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, N, F].
        Returns:
            Tensor with spatial positional encoding applied.
        """
        B, T, N, F = x.shape
        # Add spatial positional encoding
        space_encoded = self.space_pe[:, :N, :].unsqueeze(1)  # Shape: [1, 1, N, F]
        return x + space_encoded


class SpatialMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SpatialMultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: Tensor of shape [B, T, N, D]
        """
        B, T, N, D = x.shape

        Q = self.q_linear(x)  # [B, T, N, D]
        K = self.k_linear(x)  # [B, T, N, D]
        V = self.v_linear(x)  # [B, T, N, D]

        # 多头分割
        Q = Q.view(B, T, N, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, h, T, N, d_k]
        K = K.view(B, T, N, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, h, T, N, d_k]
        V = V.view(B, T, N, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, h, T, N, d_k]

        # 计算注意力分数
        attn_weights = F.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1
        )  # [B, h, T, N, N]

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)  # [B, h, T, N, d_k]

        # 合并多头输出
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(B, T, N, self.embed_dim)  # [B, T, N, D]

        # 投影回原始维度
        output = self.out_proj(attn_output)  # [B, T, N, D]
        return output


class SpatialTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super(SpatialTransformerEncoderLayer, self).__init__()
        self.self_attn = SpatialMultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Tensor of shape [B, T, N, D]
        """
        # Multi-head Self-Attention
        attn_output = self.self_attn(x)  # [B, T, N, D]
        out1 = self.norm1(x + attn_output)  # Residual connection + Norm
        out1 = self.dropout1(out1)  # Dropout after Norm

        # Feed-Forward Network
        ffn_output = self.ffn(out1)
        # Residual connection + Norm
        out2 = self.norm2(out1 + ffn_output)
        out2 = self.dropout2(out2)  # Dropout after Norm

        return out2



class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, num_layers, dropout=0.1):
        super(SpatialTransformer, self).__init__()
        self.spital_pe = SpatialPositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            SpatialTransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: Tensor of shape [B, T, N, D]
        """
        x = self.spital_pe(x)
        for layer in self.layers:
            x = layer(x)
        return x

# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, h, adj):
#         Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
#         e = self._prepare_attentional_mechanism_input(Wh)
#         zero_vec = -9e15 * torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=-1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         # attention——> [B, N, N]
#         h_prime = torch.matmul(attention, Wh)
#
#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
#
#     def _prepare_attentional_mechanism_input(self, Wh):
#         Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
#         Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
#         # broadcast add
#         e = Wh1 + Wh2.transpose(2, 3)
#         return self.leakyrelu(e)
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
#
#
# class GAT(nn.Module):
#     def __init__(self, n_in, n_out, dropout, alpha, nheads, order=1):
#         """Dense version of GAT."""
#         super(GAT, self).__init__()
#         self.dropout = dropout
#         self.nheads = nheads
#         self.order = order
#
#         self.attentions = [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
#                            range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#         for k in range(2, self.order + 1):
#             self.attentions_2 = ModuleList(
#                 [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
#                  range(nheads)])
#
#         self.out_att = GraphAttentionLayer(n_out * nheads * order, n_out, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         for k in range(2, self.order + 1):
#             x2 = torch.cat([att(x, adj) for att in self.attentions_2], dim=-1)
#             x = torch.cat([x, x2], dim=-1)
#         x = F.elu(self.out_att(x, adj))
#         return x


class SpitalBlock(nn.Module):
    def __init__(self, device, embed_dim, num_heads, ffn_dim, num_layers, dropout, Spital_Transformer_bool, GAT_bool):
        super(SpitalBlock, self).__init__()
        self.Spital_Transformer_bool = Spital_Transformer_bool
        self.GAT_bool = GAT_bool
        self.SpitalTransformer = SpatialTransformer(embed_dim, num_heads, ffn_dim, num_layers, dropout)
        # self.GAT_Layer = GAT(embed_dim, embed_dim, dropout, alpha=0.01, nheads=1, order=1)
        self.GAT_Layer = GAT_Layer(device, embed_dim, embed_dim, edge_dim=1)

        self.fs = nn.Linear(embed_dim, embed_dim)
        self.fg = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, adj):
        """
        :param x: [B, T, N, d]
        :return: [B, T, N, d]
        """
        if self.Spital_Transformer_bool and self.GAT_bool:
            out_S = self.SpitalTransformer(x)
            out_G = self.GAT_Layer(x, adj)
            gate =torch.sigmoid(self.fs(out_S) + self.fg(out_G))
            out = gate * out_S + (1 - gate) * out_G
        elif self.Spital_Transformer_bool:
            out = self.SpitalTransformer(x)
        elif self.GAT_bool:
            out = self.GAT_Layer(x, adj)
        else:
            out = x
        return out


class PredictionLayer(nn.Module):
    def __init__(self, T_dim, output_T_dim, embed_size):
        super(PredictionLayer, self).__init__()

        # 缩小时间维度。
        self.conv1 = nn.Conv2d(T_dim, output_T_dim, 1)
        # 缩小通道数，降到1维。
        self.conv2 = nn.Conv2d(embed_size, 1, 1)
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


class model_2(nn.Module):
    def __init__(self, adj, device, in_channels, hidden_size, num_heads, dropout, forward_expansion, T_in, T_out,recent_only, spital_transformer_bool, GAT_bool, temporal_transformer_bool):
        super(model_2, self).__init__()
        self.adj = adj
        self.temporal_transformer_bool = temporal_transformer_bool
        self.start_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=(1, 1))
        self.temporal_conv = Temporal_Conv_Layer(in_channels=hidden_size, out_channels=hidden_size, recent_only=recent_only)
        # self.spital_transformer = SpatialTransformer(embed_dim=hidden_size, num_heads=num_heads, ffn_dim=hidden_size * forward_expansion, num_layers=1, dropout=dropout)
        self.spital_block = SpitalBlock(device=device, embed_dim=hidden_size, num_heads=num_heads, ffn_dim=hidden_size * forward_expansion, num_layers=1, dropout=dropout, Spital_Transformer_bool=spital_transformer_bool, GAT_bool=GAT_bool)
        self.temporal_transformer = TemporalTransformer(embed_dim=hidden_size, num_heads=num_heads, ffn_dim=hidden_size * forward_expansion, num_layers=1, dropout=dropout)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x_recent, x_day, x_week):
        """
        :param x_recent: [B, in_channels, N, T_in]
        :param x_day: [B, in_channels, N, T_in]
        :param x_week: [B, in_channels, N, T_in]
        :return: [B, N, T_out]
        """
        x_recent = self.start_conv(x_recent)
        x_day = self.start_conv(x_day)
        x_week = self.start_conv(x_week)
        x = self.temporal_conv(x_recent, x_day, x_week)  # [B, D, N, T]
        x = x.permute(0, 3, 2, 1)  # [B, T, N, D]
        # x = self.spital_transformer(x)
        x = self.spital_block(x, self.adj)
        if self.temporal_transformer_bool:
            x = self.temporal_transformer(x)
        x = self.prediction_layer(x)  # 预测层输入维度为[B, T, N, D]
        return x