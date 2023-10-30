import torch
import torch.nn as nn 
import torch.nn.functional as F 

class FNN(nn.Module):
    '''
    多層パーセプトロン
    '''
    def __init__(self, input_dim, modal):
        super().__init__() 

        unimodal = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU() ,
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 3),
        )

        bimodal = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU() ,
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU() ,
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU() ,
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 3),
        )

        trimodal = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU() ,
            nn.Dropout(0.3),
            nn.Linear(192, 64),
            nn.ReLU() ,
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 3),
        )

        if len(modal) == 3:
            self.stack = trimodal
        elif len(modal) == 2:
            self.stack = bimodal 
        else:
            self.stack = unimodal

    def forward(self, x):
        x = self.stack(x) 
        y = F.softmax(x, dim=1)
        return y

class FNNUniModel(nn.Module):
    def __init__(self, input_dim):
        super(FNNUniModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = F.relu(h)
        h = self.dropout(h)
        y = self.fc3(h)
        # y = torch.sigmoid(h)
        
        return y

class FNNTernaryModel(nn.Module):
    def __init__(self, input_dim):
        super(FNNTernaryModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = F.relu(h)
        h = self.dropout(h)
        y = self.fc3(h)
        return y

class GRUModel(nn.Module):
    def __init__(self, input_dim):
        super(GRUModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.dropout(h)
        h, _ = self.gru(h)
        h = self.fc2(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc3(h)
        y = F.sigmoid(h)
        
        return y

class NN(nn.Module):
    def __init__(self, input_size=768, hidden_size=100, output_size=2):
        super(NN, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, hidden_size)
        self.l4 = torch.nn.Linear(hidden_size, hidden_size)
        self.l5 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.nn.functional.relu(self.l1(x))
        h = torch.nn.functional.relu(self.l2(h))
        h = torch.nn.functional.relu(self.l3(h))
        h = torch.nn.functional.relu(self.l4(h))
        out = torch.nn.functional.sigmoid(self.l5(h))
        return out

class MTL_NN(nn.Module):
    def __init__(self, input_size=768, hidden_size=100, output_size=2, dropout=0.25):
        super(MTL_NN, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, hidden_size)

        self.ss1 = torch.nn.Linear(hidden_size, hidden_size)
        self.ss2 = torch.nn.Linear(hidden_size, output_size)

        self.ts1 = torch.nn.Linear(hidden_size, hidden_size)
        self.ts2 = torch.nn.Linear(hidden_size, output_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        h = torch.nn.functional.relu(self.l1(x))
        h = self.dropout(h)
        h = torch.nn.functional.relu(self.l2(h))
        h = self.dropout(h)
        h = torch.nn.functional.relu(self.l3(h))
        h = self.dropout(h)

        ss_h = torch.nn.functional.relu(self.ss1(h))
        ss_h = self.dropout(ss_h)
        # ss_out = torch.nn.functional.sigmoid(self.ss2(ss_h))
        ss_out = self.ss2(ss_h)

        ts_h = torch.nn.functional.relu(self.ts1(h))
        ts_h = self.dropout(ts_h)
        # ts_out = torch.nn.functional.sigmoid(self.ts2(ts_h))
        ts_out = self.ts2(ts_h)
        return ss_out, ts_out

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

        # if config["persona"] == 'e':
        #     self.persona = 0 
        # if config["persona"] == 'a':
        #     self.persona = 1 
        # if config["persona"] == 'c':
        #     self.persona = 2 
        # if config["persona"] == 'n':
        #     self.persona = 3 
        # if config["persona"] == 'o':
        #     self.persona = 4 
        

    def forward(self, text, audio, visual):
        text_h, _ = self.textlstm(text)
        audio_h, _ = self.audiolstm(audio)
        visual_h, _ = self.visuallstm(visual)

        # persona = persona[:, :, self.persona]
        # persona = persona.view(-1, text.shape[1], 1)

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



