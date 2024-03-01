import torch
import torch.nn as nn 
import torch.nn.functional as F 

class FNN(nn.Module):
    def __init__(self, input_dim):
        super(FNN, self).__init__()

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
