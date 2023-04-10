import torch.nn as nn 
import torch.nn.functional as F 

class LSTMModel(nn.Module):
    """マルチタスク用LSTMモデル

    心象ラベルとしてsentiment(7段階)を使用。誤差関数はMSELossを想定。
    """
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        D_h1 = config['D_h1'] 
        D_h2 = config['D_h2']
        dropout = config['dropout']

        self.lstm = nn.LSTM(input_size=1218, hidden_size=D_h1, batch_first=True)
        self.linear = nn.Linear(D_h1, D_h2) # linear for sentiment
        self.plinear = nn.Linear(D_h2, 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = F.relu(self.linear(out[:, -1, :]))
        h = self.dropout(h)
        h = self.plinear(h)
        # y = F.softmax(h, dim=1)

        return h

class LSTMMultitaskModel(nn.Module):
    """マルチタスク用LSTMモデル

    心象ラベルとしてsentiment(7段階)を使用。誤差関数はMSELossを想定。
    """
    def __init__(self, config):
        super(LSTMMultitaskModel, self).__init__()
        D_h1 = config['D_h1']
        D_h2 = config['D_h2']
        dropout = config['dropout']

        self.lstm = nn.LSTM(1218, hidden_size=D_h1, batch_first=True)
        self.slinear1 = nn.Linear(D_h1, D_h2) # linear for sentiment
        self.slinear2 = nn.Linear(D_h2, 3)
        self.plinear1 = nn.Linear(D_h1, D_h2)
        self.plinear2 = nn.Linear(D_h2, 5) # linear for personality trait
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)

        # 心象推定
        hs = F.relu(self.slinear1(out))
        hs = self.dropout(hs)
        ys = self.slinear2(hs)

        # 性格特性推定
        hp = F.relu(self.plinear1(out[:, -1, :]))
        hp = self.dropout(hp)
        hp = self.plinear2(hp)
        yp = self.sigmoid(hp)

        return yp, ys


class LSTMSentimentModel(nn.Module):

    def __init__(self, config):
        
        super(LSTMSentimentModel, self).__init__()
        D_h1 = config['D_h1']
        D_h2 = config['D_h2']
        dropout = config['dropout']

        self.lstm = nn.LSTM(input_size=1218, hidden_size=D_h1, batch_first=True)
        self.linear = nn.Linear(D_h1, D_h2)
        self.slinear = nn.Linear(D_h2, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = F.relu(self.linear(out))
        h = self.dropout(h)
        y = self.slinear(h)

        return y

# class LSTMModel(nn.Module):

#     def __init__(self, D_i, D_h, args=None):
        
#         super(LSTMModel, self).__init__()
#         self.hidden_size = D_h
#         self.args = args
#         # self.dropout = nn.Dropout(dropout)

#         self.lstm = nn.LSTM(input_size=D_i, hidden_size=D_h,  batch_first=True)
#         self.linear1 = nn.Linear(D_h, D_h) 
#         self.linear2 = nn.Linear(D_h, 5)
#         self.sigmoid = nn.Sigmoid()

#         self.attention = nn.MultiheadAttention(embed_dim=D_h, num_heads=1, batch_first=True)


#     def forward(self, x):

#         out, hc = self.lstm(x) 
#         bsz = out.size()[0]
#         seq_len = out.size()[1]

#         if self.args.pred: # pooling on prediction
#             h = self.linear1(out) 
#             if self.args.attn != 0:
#                 h, _ = self.attention(h, h, h)
#             h = self.linear2(h)
#             h = self.sigmoid(h)
            
#             h_avg = F.adaptive_avg_pool2d(
#                 h.view(bsz, 1, seq_len, -1),
#                 (1, 5)
#             ).squeeze(1)

#             h_max = F.adaptive_max_pool2d(
#                 h.view(bsz, 1, seq_len, -1), 
#                 (1, 5) 
#             ).squeeze(1)

#             if self.args.pooling_type == 0:
#                 y = h[:, -1, :]
#             elif self.args.pooling_type == 1:
#                 y = h_avg.squeeze(1)
#             else:
#                 y = h_max.squeeze(1)


#         else: # pooling on feature

#             h_avg = F.adaptive_avg_pool2d(
#                 out.view(bsz, 1, seq_len, -1),
#                 (1, self.hidden_size)
#             ).squeeze(1)

#             h_max = F.adaptive_max_pool2d(
#                 out.view(bsz, 1, seq_len, -1), 
#                 (1, self.hidden_size) 
#             ).squeeze(1)

#             # hc = torch.cat([hc[0][0], h_avg.squeeze(1)], dim=-1)

#             if self.args.pooling_type == 0:
#                 h = hc[0][0] 
#             elif self.args.pooling_type == 1:
#                 h = h_avg.squeeze(1) 
#             else:
#                 h = h_max.squeeze(1)
            
#             h = self.linear1(h) # hc[0][0] or h_avg.squeeze(1), h_max.squeeze(1)
#             h = self.linear2(h)
#             y = self.sigmoid(h)

#         return y


# class biLSTMModel(nn.Module):

#     def __init__(self, D_i, D_h, D_o, n_classes=5, dropout=0.5):
        
#         super(biLSTMModel, self).__init__()

#         self.dropout = nn.Dropout(dropout)
#         self.lstm = nn.LSTM(input_size=D_i, hidden_size=D_h, bidirectional=True, batch_first=True)

#         self.linear1 = nn.Linear(D_h*2, D_o)
#         self.linear2 = nn.Linear(D_o, n_classes)

#     def forward(self, x):
#         _, hc = self.lstm(x)
#         out = torch.cat([hc[0][0], hc[0][1]], dim=1)
#         h = F.relu(self.linear1(out))
#         h = self.dropout(h)
#         y = self.linear2(h)
#         return y


# class LSTMMultitaskModel(nn.Module):
#     """マルチタスク用LSTMモデル

#     心象ラベルとしてsentiment(7段階)を使用。誤差関数はMSELossを想定。
#     """
#     def __init__(self, D_i, D_h, args=None):
#         super(LSTMMultitaskModel, self).__init__()
#         self.hidden_size = D_h
#         self.args = args
#         # self.dropout = nn.Dropout(dropout)

#         self.lstm = nn.LSTM(input_size=D_i, hidden_size=D_h, batch_first=True)
#         self.slinear1 = nn.Linear(D_h, 128) # linear for sentiment
#         self.slinear2 = nn.Linear(128, 3)
#         self.plinear1 = nn.Linear(D_h, 128)
#         self.plinear2 = nn.Linear(128, 5) # linear for personality trait
#         self.sigmoid = nn.Sigmoid()

#         self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=1, batch_first=True)


#     def forward(self, x):
#         out, _ = self.lstm(x)
#         bsz = out.size()[0]
#         seq_len = out.size()[1]
        
#         if self.args.pred: # pooling on prediction
#             # h_persona, _ = self.attention(out, out, out)
#             hs = self.slinear1(out) 
#             hp = self.plinear1(out)
#             if self.args.attn == 1:
#                 hp, _  = self.attention(hp, hp, hp)
#             elif self.args.attn == 2:
#                 hp, _ = self.attention(hp, hs, hs)

#             hp = self.plinear2(hp)            
#             hp = self.sigmoid(hp)

#             ys = self.slinear2(hs)

#             hp_avg = F.adaptive_avg_pool2d(
#                 hp.view(bsz, 1, seq_len, -1),
#                 (1, 5)
#             ).squeeze(1)

#             hp_max = F.adaptive_max_pool2d(
#                 hp.view(bsz, 1, seq_len, -1), 
#                 (1, 5) 
#             ).squeeze(1)

#             if self.args.pooling_type == 0:
#                 yp = hp[:, -1, :]
#             elif self.args.pooling_type == 1:
#                 yp = hp_avg.squeeze(1) 
#             else:
#                 yp = hp_max.squeeze(1)

#         else: # pooling on feture

#             hp_avg = F.adaptive_avg_pool2d(
#                 out.view(bsz, 1, seq_len, -1),
#                 (1, self.hidden_size)
#             ).squeeze(1)

#             hp_max = F.adaptive_max_pool2d(
#                 out.view(bsz, 1, seq_len, -1), 
#                 (1, self.hidden_size) 
#             ).squeeze(1)

#             # hc = torch.cat([h_max.squeeze(1)], dim=-1)
#             if self.args.pooling_type == 0:
#                 hp = out[:, -1, :] 
#             elif self.args.pooling_type == 1:
#                 hp = hp_avg.squeeze(1)
#             else:
#                 hp = hp_max.squeeze(1)

#             hp = self.plinear1(hp)
#             hs = self.slinear1(out)
#             if self.args.attn != 0:
#                 hp = hp.unsqueeze(dim=1)
#                 hp, _ = self.attention(hp, hs, hs)
#                 hp = hp.squeeze(1) 
#             hp = self.plinear2(hp)
#             yp = self.sigmoid(hp)
#             ys = self.slinear2(hs)

#         return yp, ys


# class biLSTMMultitaskModel(nn.Module):
#     """マルチタスク用LSTMモデル

#     心象ラベルとしてsentiment(7段階)を使用。誤差関数はMSELossを想定。
#     """

#     def __init__(self, config):
#         super(biLSTMMultitaskModel, self).__init__()
#         D_h1 = config['D_h1']
#         D_h2 = config['D_h2']
#         dropout = config['dropout']

#         self.lstm = nn.LSTM(input_size=1218, hidden_size=D_h1, bidirectional=True, batch_first=True)
#         self.plinear1 = nn.Linear(D_h1*2, D_h2)
#         self.plinear2 = nn.Linear(D_h2, 5)
#         self.slinear1 = nn.Linear(D_h1*2, D_h2)
#         self.slinear2 = nn.Linear(D_h2, 3)
#         self.sigmoid = nn.Sigmoid() 
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         out, hc = self.lstm(x)

#         hc = torch.cat([hc[0][0], hc[0][1]], dim=-1)

#         hp = F.relu(self.plinear1(hc))
#         hs = F.relu(self.slinear1(out))
#         hp = self.dropout(hp)
#         hs = self.dropout(hs)
#         hp = self.plinear2(hp)
#         ys = self.slinear2(hs)
#         yp = self.sigmoid(hp)

#         return yp, ys

# class FNNModel(nn.Module):

#     def __init__(self, D_i, D_h, D_o, n_classes = 5, dropout=0.5):
#         super(FNNModel, self).__init__() 
#         self.fc1 = nn.Linear(D_i, D_h) 
#         self.fc2 = nn.Linear(D_h, D_o) 
#         self.fc3 = nn.Linear(D_o, n_classes)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         h  = F.relu(self.fc1(x)) 
#         h = self.dropout(h)
#         h = F.relu(self.fc2(h)) 
#         h = self.dropout(h)
#         y = self.fc3(h)

#         return y


# class biLSTMSentimentModel(nn.Module):

#     def __init__(self, D_i, D_h, D_o, n_classes=5, dropout=0.5):
        
#         super(biLSTMSentimentModel, self).__init__()

#         self.dropout = nn.Dropout(dropout)
#         self.lstm = nn.LSTM(input_size=D_i, hidden_size=D_h, bidirectional=True, batch_first=True)

#         self.linear1 = nn.Linear(D_h*2, D_o)
#         self.linear2 = nn.Linear(D_o, n_classes)


#     def forward(self, x):
#         h, _ = self.lstm(x)
#         h = F.relu(self.linear1(h))
#         h = self.dropout(h)
#         y = self.linear2(h)
#         return y







