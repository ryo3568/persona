import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable 

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor


class LSTMModel(nn.Module):

    def __init__(self, D_i, D_h, D_o, n_classes=5, dropout=0.5):
        
        super(LSTMModel, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_i, hidden_size=D_h,  batch_first=True)

        self.linear1 = nn.Linear(D_h, D_o)
        self.linear2 = nn.Linear(D_o, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = F.relu(self.linear1(out[:, -1]))
        h = self.dropout(h)
        y = self.linear2(h)
        return y


class biLSTMModel(nn.Module):

    def __init__(self, D_i, D_h, D_o, n_classes=5, dropout=0.5):
        
        super(biLSTMModel, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_i, hidden_size=D_h, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(D_h*2, D_o)
        self.linear2 = nn.Linear(D_o, n_classes)

    def forward(self, x):
        _, hc = self.lstm(x)
        out = torch.cat([hc[0][0], hc[0][1]], dim=1)
        h = F.relu(self.linear1(out))
        h = self.dropout(h)
        y = self.linear2(h)
        return y


class LSTMSentimentModel(nn.Module):

    def __init__(self, D_i, D_h, D_o, n_classes=5, dropout=0.5):
        
        super(LSTMSentimentModel, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_i, hidden_size=D_h, batch_first=True)

        self.linear1 = nn.Linear(D_h, D_o)
        self.linear2 = nn.Linear(D_o, n_classes)

    def forward(self, x):
        h, _ = self.lstm(x)
        h = F.relu(self.linear1(h))
        h = self.dropout(h)
        y = self.linear2(h)
        return y

class biLSTMSentimentModel(nn.Module):

    def __init__(self, D_i, D_h, D_o, n_classes=5, dropout=0.5):
        
        super(biLSTMSentimentModel, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_i, hidden_size=D_h, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(D_h*2, D_o)
        self.linear2 = nn.Linear(D_o, n_classes)

    def forward(self, x):
        h, _ = self.lstm(x)
        h = F.relu(self.linear1(h))
        h = self.dropout(h)
        y = self.linear2(h)
        return y


class LSTMMultiTaskModel(nn.Module):
    """マルチタスク用LSTMモデル

    心象ラベルとしてsentiment(7段階)を使用。誤差関数はMSELossを想定。
    """

    def __init__(self, D_i, D_h, D_o, n_classes=3, dropout=0.5):
        super(LSTMMultiTaskModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_i, hidden_size=D_h, batch_first=True)

        self.linear = nn.Linear(D_h, D_o)
        self.linear_persona = nn.Linear(D_o, 5)
        self.linear_sentiment = nn.Linear(D_o, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        h_persona = F.relu(self.linear(out[:, -1]))
        h_sentiment = F.relu(self.linear(out))
        h_persona = self.dropout(h_persona)
        h_sentiment = self.dropout(h_sentiment)
        y_persona = self.linear_persona(h_persona)
        y_sentiment = self.linear_sentiment(h_sentiment)

        return y_persona, y_sentiment

class biLSTMMultiTaskModel(nn.Module):
    """マルチタスク用LSTMモデル

    心象ラベルとしてsentiment(7段階)を使用。誤差関数はMSELossを想定。
    """

    def __init__(self, D_i, D_h, D_o, n_classes=3, dropout=0.5):
        super(biLSTMMultiTaskModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_i, hidden_size=D_h, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(D_h*2, D_o)
        self.linear_persona = nn.Linear(D_o, 5)
        self.linear_sentiment = nn.Linear(D_o, n_classes)

    def forward(self, x):
        out, hc = self.lstm(x)
        hc = torch.cat([hc[0][0], hc[0][1]], dim=1)
        h_persona = F.relu(self.linear(hc))
        h_sentiment = F.relu(self.linear(out))
        h_persona = self.dropout(h_persona)
        h_sentiment = self.dropout(h_sentiment)
        y_persona = self.linear_persona(h_persona)
        y_sentiment = self.linear_sentiment(h_sentiment)

        return y_persona, y_sentiment

class FNNModel(nn.Module):

    def __init__(self, D_i, D_h, D_o, n_classes = 5, dropout=0.5):
        super(FNNModel, self).__init__() 
        self.fc1 = nn.Linear(D_i, D_h) 
        self.fc2 = nn.Linear(D_h, D_o) 
        self.fc3 = nn.Linear(D_o, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h  = F.relu(self.fc1(x)) 
        h = self.dropout(h)
        h = F.relu(self.fc2(h)) 
        h = self.dropout(h)
        y = self.fc3(h)

        return y

class FNNMultitaskModel(nn.Module):

    def __init__(self, D_i, D_h, D_o, n_classes = 5, dropout=0.5):
        super(FNNMultitaskModel, self).__init__() 
        self.fc1 = nn.Linear(D_i, D_h) 
        self.fc2 = nn.Linear(D_h, D_o) 
        self.fc_persona = nn.Linear(D_o, 5)
        self.fc_sentiment = nn.Linear(D_o, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h  = F.relu(self.fc1(x)) 
        h = self.dropout(h)
        h = F.relu(self.fc2(h)) 
        h = self.dropout(h)
        y_persona = self.fc_persona(h) 
        y_sentiment = self.fc_sentiment(h)

        return y_persona, y_sentiment




