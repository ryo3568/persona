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

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss

class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred*mask, target)/torch.sum(mask)
        return loss


class LSTMModel(nn.Module):

    def __init__(self, D_i, D_h, D_o, n_classes=5, dropout=0.5):
        
        super(LSTMModel, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_i, hidden_size=D_h, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(D_h*2, D_o)
        self.linear2 = nn.Linear(D_o, n_classes)

    def forward(self, x):
        h, _ = self.lstm(x)
        h = F.relu(self.linear1(h[:, -1]))
        h = self.dropout(h)
        y = self.linear2(h)
        return y

class LSTMSentimentModel(nn.Module):

    def __init__(self, D_i, D_h, D_o, n_classes=3, dropout=0.5):
        super(LSTMSentimentModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_i, hidden_size=D_h, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(D_h*2, D_o)
        self.linear2 = nn.Linear(D_o, n_classes)

    def forward(self, x):
        h, _ = self.lstm(x)
        h = F.relu(self.linear1(h))
        h = self.dropout(h)
        y = self.linear2(h)

        return y

class LSTMSentimentModel2(nn.Module):

    def __init__(self, D_i, D_h, D_o, n_classes=3, dropout=0.5):
        super(LSTMSentimentModel2, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_i, hidden_size=D_h, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)

        self.linear = nn.Linear(D_h*2, D_o)
        self.smax_fc = nn.Linear(D_o, n_classes)

    def forward(self, x):
        h, _ = self.lstm(x)
        h = F.relu(self.linear(h))
        h = self.dropout(h)
        log_prob = F.log_softmax(self.smax_fc(h), 2)

        return log_prob