import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from GNN.GNN import ChebConv_Layer, GAT_Layer

from _Support.CBAM import CBAM
from _Support.TemporalConvNet import TemporalConvNet
from _Support.convLSTM import ConvLSTM

from model_ST.STARes_STCN import STARes
from model_ST.STARes_SaLSTM import MultiHeadTemporalSelfAttention
from model.AHSTGNN import GAT, gated_TCN, ComputeAttentionScore
from GNN.GCNLayer import GCNLayer

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
    
class LSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, dropout, T_in, T_out):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(12, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, 12)
        self.T_out = T_out

    def forward(self, x):
        out, _ = self.lstm(x[:, :, :, 0].transpose(1, 2))
        out = self.linear1(out)
        return out.transpose(1, 2)[:, :, -self.T_out:]

class CNN_LSTM(nn.Module):
    def __init__(
        self,
        input_features=6,    # F: 每个站点的特征数（论文：6个污染物）
        input_steps=6,       # T: 输入时间步（论文：6小时）
        node_num=12,         # N: 站点数量（论文：12个北京站点）
        out_steps=6,         # T_out: 预测时间步
        cnn_out=16,          # CNN 输出通道
        lstm_hidden=64,      # LSTM 隐藏层维度
        lstm_layers=1
    ):
        super().__init__()
        self.T = input_steps
        self.N = node_num
        self.F = input_features
        self.T_out = out_steps

        # ===================== TimeDistributed CNN =====================
        # 对每个站点、每个时间步做特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_features,
                out_channels=cnn_out,
                kernel_size=1
            ),
            nn.ReLU()
        )

        # 计算 CNN 展平维度
        with torch.no_grad():
            dummy = torch.randn(1, input_features, input_steps)
            cnn_out_dim = self.cnn(dummy).numel()
        self.cnn_out_dim = cnn_out_dim

        # ===================== LSTM 时序建模 =====================
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        # ===================== 输出层（每个站点预测 T_out 步） =====================
        self.fc = nn.Linear(lstm_hidden, out_steps)

    def forward(self, x):
        """
        输入：x → [B, N, T, F]
        输出：y → [B, N, T_out]
        """
        B, N, T, F = x.shape

        # -------------------- 维度转换，适配 CNN 输入 --------------------
        # [B, N, T, F] → [B*N, F, T]
        x = x.permute(0, 1, 3, 2).reshape(B * N, F, T)

        # -------------------- TimeDistributed CNN --------------------
        c_out = self.cnn(x)  # [B*N, cnn_out, T]

        # 展平 + 恢复时间维度
        c_flat = c_out.flatten(1)  # [B*N, cnn_out_dim]
        c_seq = c_flat.reshape(B, N, -1)  # [B, N, cnn_out_dim]

        # -------------------- LSTM 每个站点独立时序建模 --------------------
        lstm_out, _ = self.lstm(c_seq)  # [B, N, lstm_hidden]

        # -------------------- 输出预测 --------------------
        out = self.fc(lstm_out)  # [B, N, T_out]

        return out

class TCN(nn.Module):
    def __init__(self, in_channels, num_channels, T_in, T_out):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=12, num_channels=num_channels)
        self.linear = nn.Linear(num_channels[-1], 12)
        self.T_out = T_out

    def forward(self, x):
        out = self.tcn(x[:, :, :, 0])
        out = self.linear(out.transpose(1, 2))
        return out.transpose(1, 2)[:, :, -self.T_out:]    

    
class GRU(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, dropout, T_in, T_out):
        super(GRU, self).__init__()
        self.gru = nn.GRU(12, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, 12)
        self.T_out = T_out

    def forward(self, x):
        out, _ = self.gru(x[:, :, :, 0].transpose(1, 2))
        out = self.linear1(out)
        return out.transpose(1, 2)[:, :, -self.T_out:]


class Conv_LSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, T_in, T_out):
        super(Conv_LSTM, self).__init__()
        self.linear1 = nn.Linear(in_features=in_channels, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=1, out_features=hidden_size)
        self.convLSTM = ConvLSTM(input_dim=hidden_size, hidden_dim=hidden_size, kernel_size=(3, 3),
                                 num_layers=num_layers, batch_first=True, bias=True, return_all_layers=False)
        self.linear3 = nn.Linear(in_features=hidden_size, out_features=1)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x):
        out = self.linear1(x)  # [B, N, t, d]
        out = out.permute(0, 2, 1, 3).unsqueeze(-1)  # [B, T, N, d, 1]
        out = out.permute(0, 1, 3, 2, 4)  # [B, T, d, N, 1]
        out, _ = self.convLSTM(out)  # 输出形状 [B, T, d, N, 1]
        out = out[-1].squeeze(-1)  # [B, T, d, N]
        out = out.permute(0, 1, 3, 2)
        out = self.prediction_layer(out)
        return out


class STA_ResConvLSTM(nn.Module):
    def __init__(self, input_size, input_len, hidden_size, ratio, num_layers, dropout, predict_len, STARes_blocks):
        super(STA_ResConvLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.predict_len = predict_len
        self.dropout = nn.Dropout(p=dropout)
        self.sta_res_BlockList = nn.ModuleList([
            STARes(input_len, ratio)
            for _ in range(STARes_blocks)
        ])
        self.convlstm = ConvLSTM(input_dim=input_size, hidden_dim=[hidden_size] * num_layers, kernel_size=(3, 3),
                                 num_layers=num_layers, batch_first=True, bias=True, return_all_layers=False)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, num_nodes, seq_len, features = x.size()
        x = x.permute(0, 2, 1, 3)
        for i in range(len(self.sta_res_BlockList)):
            x = self.sta_res_BlockList[i](x)
        x = x.permute(0, 1, 3, 2).unsqueeze(-1)
        x = self.dropout(x)
        output, _ = self.convlstm(x)
        x = output[-1].squeeze(-1).permute(0, 3, 1, 2)
        x = self.dropout(x)
        x = x.contiguous().view(batch_size * num_nodes, seq_len, self.hidden_size)
        x = self.linear(x)
        x = x.contiguous().view(batch_size, num_nodes, seq_len, 1)
        return x[:, :, -self.predict_len:, :]



# class CNN_LSTM(nn.Module):
#     def __init__(self, in_channels, hidden_size, num_layers, dropout, T_in, T_out):
#         super(CNN_LSTM, self).__init__()
#         self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
#         self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

#     def forward(self, x):
#         B, N, t, d = x.size()
#         x = x.permute(0, 3, 1, 2)
#         x = self.cnn(x)
#         x = x.permute(0, 2, 3, 1)
#         x = x.contiguous().view(B*N, t, -1)
#         out, _ = self.lstm(x)
#         out = out.view(B, N, t, -1)
#         out = out.permute(0, 2, 1, 3)
#         out = self.prediction_layer(out)
#         return out
    

class ChebGCN_LSTM(nn.Module):
    def __init__(self, adj, in_channels, hidden_size, K, device, num_layers, dropout, T_in, T_out):
        super(ChebGCN_LSTM, self).__init__()
        self.cheb_gcn = ChebConv_Layer(device, adj, in_channels, hidden_size, K)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x):
        """
        :param x: [B, N, T_in, D]
        :return: [B, N, T_out]
        """
        B, N, t, d = x.size()
        out = x.permute(0, 2, 1, 3)
        out = out.reshape(B*t, N, d)
        out = self.cheb_gcn(out)
        out = out.reshape(B, t, N, -1)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B*N, t, -1)
        out = F.relu(out)
        out, _ = self.lstm(out)
        out = out.reshape(B, N, t, -1)
        out = out.permute(0, 2, 1, 3)
        out = self.prediction_layer(out)
        return out

class GAT_LSTM(nn.Module):
    def __init__(self, adj, in_channels, hidden_size, device, num_layers, dropout, T_in, T_out):
        super(GAT_LSTM, self).__init__()
        self.GAT1 = GAT_Layer(device, adj, in_channels, hidden_size, edge_dim=1)
        self.GAT2 = GAT_Layer(device, adj, hidden_size, hidden_size, edge_dim=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x):
        """
        :param x: [B, N, T_in, D]
        :return: [B, N, T_out]
        """
        B, N, t, d = x.size()
        out = x.permute(0, 2, 1, 3)
        out = out.reshape(B*t, N, d)
        out = self.GAT1(out)
        out = F.relu(out)
        out = self.GAT2(out)
        out = F.relu(out)
        out = out.reshape(B, t, N, -1)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B*N, t, -1)
        out, _ = self.lstm(out)
        out = out.reshape(B, N, t, -1)
        out = out.permute(0, 2, 1, 3)
        out = self.prediction_layer(out)
        return out


class MSTGCN_block(nn.Module):
    def __init__(self, input_dim, hidden_size, K, edge_index, edge_weight):
        super(MSTGCN_block, self).__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.gcn = ChebConv(in_channels=input_dim, out_channels=hidden_size, K=K)
        self.temporal_conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels=input_dim, out_channels=hidden_size, kernel_size=(1, 1),
                                       stride=(1, 1))

    def forward(self, input):
        x = input.transpose(1, 2)
        batch_size, seq_len, num_nodes, features = x.size()
        x = x.contiguous().view(batch_size * seq_len, num_nodes, features)
        x = self.gcn(x, edge_index=self.edge_index, edge_weight=self.edge_weight)
        x = x.contiguous().view(batch_size, seq_len, num_nodes, self.hidden_size)
        x = x.permute(0, 3, 2, 1)
        time_conv_output = self.temporal_conv(x)
        x_residual = self.residual_conv(input.permute(0, 3, 1, 2))
        x = (time_conv_output + x_residual).permute(0, 2, 3, 1)
        return x



class MSTGCN(nn.Module):
    def __init__(self, input_dim, hidden_size, K, predict_len, block_num, edge_index, edge_weight):
        super(MSTGCN, self).__init__()
        self.predict_len = predict_len
        self.hidden_size = hidden_size
        self.Blocklist = nn.ModuleList(
            [MSTGCN_block(input_dim, hidden_size, K, edge_index, edge_weight) if i == 0 else
             MSTGCN_block(hidden_size, hidden_size, K, edge_index, edge_weight)
             for i in range(block_num)])
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, num_nodes, seq_len, _ = x.size()
        for i in range(len(self.Blocklist)):
            x = self.Blocklist[i](x)
        x = x.contiguous().view(batch_size * num_nodes, seq_len, self.hidden_size)
        x = self.linear(x)
        x = x.contiguous().view(batch_size, num_nodes, seq_len, 1)
        return x[:, :, -self.predict_len:, :]


class GCN_TCN_STA_block(nn.Module):
    def __init__(self, input_size, hidden_size, K, num_nodes, input_len, num_channels, kernel_size, dropout, edge_index, edge_weight):
        super(GCN_TCN_STA_block, self).__init__()
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.gcn_sta = ASTGCNBlock(input_size, K, hidden_size, num_nodes, input_len)
        self.tcn = TemporalConvNet(hidden_size, num_channels, kernel_size, dropout)
        self.residual_conv = nn.Conv2d(in_channels=input_size, out_channels=num_channels[-1], kernel_size=(1, 1),
                                       stride=(1, 1))

    def forward(self, input):
        batch_size, num_nodes, seq_len, _ = input.size()
        x = input.transpose(2, 3)
        x = self.gcn_sta(x, edge_index=self.edge_index, edge_weight=self.edge_weight)
        x = x.contiguous().view(batch_size * num_nodes, self.hidden_size, seq_len)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, num_nodes, seq_len, self.num_channels[-1])
        x_residual = self.residual_conv(input.permute(0, 3, 1, 2))
        x = (x + x_residual.permute(0, 2, 3, 1))
        return x


class GCN_TCN_STA(nn.Module):
    def __init__(self, input_size, hidden_size, K, num_nodes, input_len, num_channels, kernel_size, dropout,
                 predict_len, block_num, edge_index, edge_weight):
        super(GCN_TCN_STA, self).__init__()
        self.num_channels = num_channels
        self.predict_len = predict_len
        self.Blocklist = nn.ModuleList([GCN_TCN_STA_block(input_size=input_size, hidden_size=hidden_size, K=K,
                                                            num_nodes=num_nodes, input_len=input_len,
                                                            num_channels=num_channels, kernel_size=kernel_size,
                                                            dropout=dropout, edge_index=edge_index, edge_weight=edge_weight)
                                        if i == 0 else
                                        GCN_TCN_STA_block(input_size=num_channels[-1], hidden_size=hidden_size, K=K,
                                                            num_nodes=num_nodes, input_len=input_len,
                                                            num_channels=num_channels, kernel_size=kernel_size,
                                                            dropout=dropout, edge_index=edge_index, edge_weight=edge_weight)
                                        for i in range(block_num)])
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        batch_size, num_nodes, seq_len, _ = x.size()
        for i in range(len(self.Blocklist)):
            x = self.Blocklist[i](x)
        x = x.contiguous().view(batch_size * num_nodes, seq_len, self.num_channels[-1])
        x = self.linear(x)
        x = x.contiguous().view(batch_size, num_nodes, seq_len, 1)
        return x[:, :, -self.predict_len:, :]



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
            out = self.gat(x, self.edge_index)
        return out


class M_Adaptive_Attention_STGNN_block(nn.Module):
    def __init__(self, in_channels, hidden_size, dropout, alpha, n_heads, kernel_size, layers, num_nodes, apt_size, adj, gated_TCN_bool, gcn_bool, gat_bool, ASTAM_bool):
        super(M_Adaptive_Attention_STGNN_block, self).__init__()
        # 参数
        self.gated_TCN_bool = gated_TCN_bool
        self.gcn_bool = gcn_bool
        self.gat_bool = gat_bool
        self.ASTAM_bool = ASTAM_bool
        # 扩展维度
        self.start_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=(1, 1))
        # 时间模块
        self.gated_TCN = gated_TCN(hidden_size, hidden_size, kernel_size, layers)
        self.tcn = dilated_TCN(hidden_size, hidden_size, kernel_size, layers)
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
        x = self.start_conv(x)
        if self.gated_TCN_bool:
            x_t = self.gated_TCN(x).transpose(1, 3)
        else:
            x_t = self.tcn(x).transpose(1, 3)
        x_s = self.spital_block(x)
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


class M_Adaptive_Attention_STGNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, alpha, n_heads, kernel_size, layers, apt_size, num_nodes, num_block, predict_len, gated_TCN_bool, gcn_bool, gat_bool, ASTAM_bool):
        super(M_Adaptive_Attention_STGNN, self).__init__()
        self.hidden_size = hidden_size
        self.predict_len = predict_len
        self.Blocklist = nn.ModuleList([
            M_Adaptive_Attention_STGNN_block(in_channels=input_size, hidden_size=hidden_size, dropout=dropout, alpha=alpha, n_heads=n_heads, kernel_size=kernel_size, layers=layers, num_nodes=num_nodes, apt_size=apt_size,
                                             gated_TCN_bool=gated_TCN_bool, gcn_bool=gcn_bool, gat_bool=gat_bool, ASTAM_bool=ASTAM_bool)
            if i == 0 else
            M_Adaptive_Attention_STGNN_block(in_channels=hidden_size, hidden_size=hidden_size, dropout=dropout, alpha=alpha, n_heads=n_heads, kernel_size=kernel_size, layers=layers, num_nodes=num_nodes, apt_size=apt_size,
                                             gated_TCN_bool=gated_TCN_bool, gcn_bool=gcn_bool, gat_bool=gat_bool, ASTAM_bool=ASTAM_bool)
            for i in range(num_block)
        ])
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        for i in range(len(self.Blocklist)):
            x = self.Blocklist[i](x)
        B, N, t, _ = x.size()
        x = x.contiguous().view(B * N, t, self.hidden_size)
        x = self.linear(x)
        x = x.contiguous().view(B, N, t, 1)
        return x[:, :, -self.predict_len:, :]