import torch
import torch.nn as nn 
import torch.nn.functional as F 

class LateFusionModel(nn.Module):

    def __init__(self, config, id):
        
        super(LateFusionModel, self).__init__()
        self.config = config

        self.textlstm = TextModel(config)
        self.textlstm.load_state_dict(torch.load(f"../data/model/text/{id}"))

        self.audiolstm = AudioModel(config)
        self.audiolstm.load_state_dict(torch.load(f"../data/model/audio/{id}"))

        self.visuallstm = VisualModel(config)
        self.visuallstm.load_state_dict(torch.load(f"../data/model/visual/{id}"))

        self.classifier = nn.Linear(9, 3)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, text, audio, visual):
        text_h, _ = self.textlstm(text)
        audio_h, _ = self.audiolstm(audio)
        visual_h, _ = self.visuallstm(visual)

        h = torch.cat([text_h, audio_h, visual_h], dim=-1)
        y = self.classifier(h)
        return y
    
class TextModel(nn.Module):

    def __init__(self, config):
        super(TextModel, self).__init__()
        D_h1 = config['D_h1']
        D_h2 = config['D_h2']
        # dropout = config['dropout']

        self.lstm = nn.LSTM(input_size=768, hidden_size=D_h1, batch_first=True)
        self.linear = nn.Linear(D_h1, D_h2)
        self.classifier = nn.Linear(D_h2, 3)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = F.relu(self.linear(out))
        # h = self.dropout(h)
        y = self.classifier(h)
        return y, h

class AudioModel(nn.Module):

    def __init__(self, config):
        super(AudioModel, self).__init__()
        D_h1 = config['D_h1']
        D_h2 = config['D_h2']
        # dropout = config['dropout']

        self.lstm = nn.LSTM(input_size=384, hidden_size=D_h1, batch_first=True)
        self.linear = nn.Linear(D_h1, D_h2)
        self.classifier = nn.Linear(D_h2, 3)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = F.relu(self.linear(out))
        # h = self.dropout(h)
        y = self.classifier(h)
        return y, h

class VisualModel(nn.Module):

    def __init__(self, config):
        super(VisualModel, self).__init__()
        D_h1 = config['D_h1']
        D_h2 = config['D_h2']
        # dropout = config['dropout']

        self.lstm = nn.LSTM(input_size=66, hidden_size=D_h1, batch_first=True)
        self.linear = nn.Linear(D_h1, D_h2)
        self.classifier = nn.Linear(D_h2, 3)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = F.relu(self.linear(out))
        # h = self.dropout(h)
        y = self.classifier(h)
        return y, h
    

class EarlyFusionModel(nn.Module):

    def __init__(self, config):
        
        super(EarlyFusionModel, self).__init__()
        D_h1 = config['D_h1']
        D_h2 = config['D_h2']
        dropout = config['dropout']

        self.lstm = nn.LSTM(input_size=config["input_size"], hidden_size=D_h1, batch_first=True)
        self.linear = nn.Linear(D_h1, D_h2)
        self.slinear = nn.Linear(D_h2, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = F.relu(self.linear(out))
        h = self.dropout(h)
        y = self.slinear(h)

        return y

# 不要なモデル

# class LSTMModel(nn.Module):
#     """
#     性格特性推定モデル
#     """
#     def __init__(self, config):
#         super(LSTMModel, self).__init__()
#         D_h1 = config['D_h1'] 
#         D_h2 = config['D_h2']
#         dropout = config['dropout']

#         self.lstm = nn.LSTM(input_size=1218, hidden_size=D_h1, batch_first=True)
#         self.linear = nn.Linear(D_h1, D_h2)
#         self.plinear1 = nn.Linear(D_h2, D_h2)
#         self.plinear2 = nn.Linear(D_h2, 5)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         h = F.relu(self.linear(out[:, -1, :]))
#         h = self.dropout(h)
#         h = self.plinear1(h)
#         h = self.plinear2(h)

#         return h

# class LSTMMultitaskModel(nn.Module):
#     """マルチタスク用LSTMモデル

#     心象ラベルとしてsentiment(7段階)を使用。誤差関数はMSELossを想定。
#     """
#     def __init__(self, config):
#         super(LSTMMultitaskModel, self).__init__()
#         D_h1 = config['D_h1']
#         D_h2 = config['D_h2']
#         dropout = config['dropout']

#         self.lstm = nn.LSTM(1218, hidden_size=D_h1, batch_first=True)
#         self.slinear1 = nn.Linear(D_h1, D_h2) # linear for sentiment
#         self.slinear2 = nn.Linear(D_h2, 3)
#         self.plinear1 = nn.Linear(D_h1, D_h2)
#         self.plinear2 = nn.Linear(D_h2, 5) # linear for personality trait
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         out, _ = self.lstm(x)

#         # 心象推定
#         hs = F.relu(self.slinear1(out))
#         hs = self.dropout(hs)
#         ys = self.slinear2(hs)

#         # 性格特性推定
#         hp = F.relu(self.plinear1(out[:, -1, :]))
#         hp = self.dropout(hp)
#         hp = self.plinear2(hp)
#         yp = self.sigmoid(hp)

#         return yp, ys



