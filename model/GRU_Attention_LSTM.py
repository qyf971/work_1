import torch
import torch.nn as nn
import torch.nn.functional as F

# 注意力层
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.w = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, gru_outputs):
        # gru_outputs: [batch, seq_len, hidden_size]
        x = torch.tanh(self.w(gru_outputs))   
        score = self.v(x)                     
        att_weight = F.softmax(score, dim=1)  
        context = torch.sum(att_weight * gru_outputs, dim=1) 
        return context, att_weight

# 论文核心模型：GRU + Attention + LSTM
class GRU_Attention_LSTM(nn.Module):
    def __init__(self, input_size=7, gru_hidden=64, lstm_hidden=32, output_size=1):
        super(GRU_Attention_LSTM, self).__init__()
        
        # GRU 层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )
        
        # 注意力层
        self.attention = AttentionLayer(gru_hidden)
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=gru_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        
        # 输出层
        self.fc = nn.Linear(lstm_hidden, output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        
        # GRU 前向
        gru_out, _ = self.gru(x)  
        
        # 注意力加权
        att_out, _ = self.attention(gru_out)  
        att_out = att_out.unsqueeze(1).repeat(1, gru_out.shape[1], 1)  
        
        # LSTM 提取时序
        lstm_out, _ = self.lstm(att_out)  
        
        # 预测输出
        out = self.fc(lstm_out[:, -1, :])  
        return out