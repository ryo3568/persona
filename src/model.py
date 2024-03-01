import torch.nn as nn 
import torch.nn.functional as F 

class FNN(nn.Module):
    def __init__(self, input_dim, output_dim, modal='t'):
        super(FNN, self).__init__() 
        
        if len(modal) == 1:
            k = 1
        elif len(modal) == 2:
            k = 2
        else:
            k = 3

        self.fc1 = nn.Linear(input_dim, 64 * k)
        self.fc2 = nn.Linear(64 * k, 64 * k)
        self.fc3 = nn.Linear(64 * k, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, output_dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        h = F.relu(self.fc3(h))
        h = self.dropout(h)
        h = F.relu(self.fc4(h))
        h = self.dropout(h)
        y = self.fc5(h)
        return y
    
