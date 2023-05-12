import torch
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
        self.linear = nn.Linear(D_h1, D_h2)
        self.plinear1 = nn.Linear(D_h2, D_h2)
        self.plinear2 = nn.Linear(D_h2, 5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = F.relu(self.linear(out[:, -1, :]))
        h = self.dropout(h)
        h = self.plinear1(h)
        h = self.plinear2(h)

        return h

class sLSTMearlyModel(nn.Module):

    def __init__(self, config):
        
        super(sLSTMearlyModel, self).__init__()
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

class sLSTMlateModel(nn.Module):

    def __init__(self, config):
        
        super(sLSTMlateModel, self).__init__()
        # D_h1 = config['D_h1']
        # D_h2 = config['D_h2']
        # dropout = config['dropout']
        self.config = config

        input_size = 0
        if 't' in self.config["modal"]:
            self.textlstm = textModel(config)
            input_size += 3
        if 'a' in self.config["modal"]:
            self.audiolstm = audioModel(config)
            input_size += 3
        if 'v' in self.config["modal"]:
            self.visuallstm = visualModel(config)
            input_size += 3

        self.classifier = nn.Linear(input_size, 3)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, text, audio, visual):
        input_data = []
        if 't' in self.config["modal"]:
            text_h, _ = self.textlstm(text)
            input_data.append(text_h)
        if 'a' in self.config["modal"]:
            audio_h, _ = self.audiolstm(audio)
            input_data.append(audio_h)
        if 'v' in self.config["modal"]:
            visual_h, _ = self.visuallstm(visual)
            input_data.append(visual_h)
        h = torch.cat(input_data, dim=-1)
        y = self.classifier(h)
        return y
    
class textModel(nn.Module):

    def __init__(self, config):
        super(textModel, self).__init__()
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

class audioModel(nn.Module):

    def __init__(self, config):
        super(audioModel, self).__init__()
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

class visualModel(nn.Module):

    def __init__(self, config):
        super(visualModel, self).__init__()
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
    

# 不要なモデル

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



