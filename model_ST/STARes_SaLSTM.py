import torch
import torch.nn as nn
import torch.nn.functional as F

from _Support.embed import TokenEmbedding
from model_ST.STARes_STCN import STARes

class PredictionLayer(nn.Module):
    def __init__(self, T_dim, output_T_dim, embed_size):
        super(PredictionLayer, self).__init__()

        # 缩小时间维度。
        self.conv1 = nn.Conv2d(T_dim, output_T_dim, 1)
        # 缩小通道数，降到1维。
        self.conv2 = nn.Conv2d(embed_size, 1, 1)

    def forward(self, input_prediction_layer):
        """
        :param input_prediction_layer: [B, T, N, D]
        :return: [B, N, out_T]
        """
        out = self.conv1(input_prediction_layer) # 等号左边 out shape: [B, T, N, d]
        out = out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, d, N, T]
        out = self.conv2(out)  # 等号左边 out shape: [B, 1, N, T]
        out = out.squeeze(1)
        return out


class MultiHeadTemporalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim=512):
        super(MultiHeadTemporalSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # LSTM用于生成Q、K、V
        self.lstm_q = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=False)
        self.lstm_k = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=False)
        self.lstm_v = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=False)

        # 线性层用于生成Q、K、V
        self.q_linear = nn.Linear(hidden_dim, embed_dim)
        self.k_linear = nn.Linear(hidden_dim, embed_dim)
        self.v_linear = nn.Linear(hidden_dim, embed_dim)

        # 线性层用于合并多头注意力的结果
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # 使用LSTM生成Q、K、V
        q_lstm_out, _ = self.lstm_q(x)
        k_lstm_out, _ = self.lstm_k(x)
        v_lstm_out, _ = self.lstm_v(x)

        # 线性变换生成Q、K、V
        Q = self.q_linear(q_lstm_out)
        K = self.k_linear(k_lstm_out)
        V = self.v_linear(v_lstm_out)

        # 分割成多个头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 应用softmax
        attention_weights = F.softmax(scores, dim=-1)

        # 加权求和得到输出
        context = torch.matmul(attention_weights, V)

        # 合并多个头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 最终线性层
        output = self.out_linear(context)

        return output, attention_weights


class STARes_SaLSTM(nn.Module):
    def __init__(self, input_size, input_len, ratio, embed_dim, num_heads, predict_len, STARes_blocks):
        super(STARes_SaLSTM, self).__init__()
        self.predict_len = predict_len
        self.sta_res_BlockList = nn.ModuleList([
            STARes(input_len, ratio)
            for _ in range(STARes_blocks)
        ])
        self.token_embedding = TokenEmbedding(input_size=input_size, d_model=embed_dim)
        self.multi_head_temporal_self_attention = MultiHeadTemporalSelfAttention(embed_dim, num_heads)
        self.prediction_layer = PredictionLayer(T_dim=input_len, output_T_dim=predict_len, embed_size=embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3) # [B, T, N, F]
        for i in range(len(self.sta_res_BlockList)):
            x = self.sta_res_BlockList[i](x) # [B, T, N, F]
        x = x.permute(0, 2, 1, 3)
        batch_size, num_nodes, seq_len, features = x.size()
        x = x.reshape(batch_size * num_nodes, seq_len, features)
        x = self.token_embedding(x)
        x, _ = self.multi_head_temporal_self_attention(x)
        x = x.reshape(batch_size, num_nodes, seq_len, -1).transpose(1, 2)
        x = self.prediction_layer(x)
        return x
