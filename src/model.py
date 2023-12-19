import glob
import torch
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
    

class LatefusionFNN(nn.Module):
    def __init__(self, output_dim, modal):
        super(LatefusionFNN, self).__init__() 

        self.t_model = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
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
            nn.Linear(32, output_dim),
        )

        self.a_model = nn.Sequential(
            nn.Linear(384, 64),
            nn.ReLU(),
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
            nn.Linear(32, output_dim),
        )

        self.v_model = nn.Sequential(
            nn.Linear(66, 64),
            nn.ReLU(),
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
            nn.Linear(32, output_dim),
        )

        k = len(modal)
        self.fusion_model = nn.Sequential(
            nn.Linear(output_dim * k, output_dim)
        )

        self.modal = modal

    
    def forward(self, t_x, a_x, v_x):
        h = []
        if 't' in self.modal:
            t_y = self.t_model(t_x)
            h.append(t_y)
        if 'a' in self.modal:
            a_y = self.a_model(a_x)
            h.append(a_y)
        if 'v' in self.modal:
            v_y = self.v_model(v_x)
            h.append(v_y)
        h = torch.cat(h, dim=1)
        y = self.fusion_model(h)

        return y

class LatefusionFNN_better(nn.Module):
    def __init__(self, input_dim, num_classes, ss, pmode, pgroup, modal):
        super(LatefusionFNN_better, self).__init__() 

        self.fc = nn.Linear(input_dim, num_classes)

        self.fc1 = nn.Linear(input_dim, int(input_dim / 2))
        self.fc2 = nn.Linear(int(input_dim/2), num_classes)
        self.dropout = nn.Dropout(0.3)

        self.modal = modal
    
    def forward(self, t_x, a_x, v_x):
        h = []
        if 't' in self.modal:
            h.append(t_x)
        if 'a' in self.modal:
            h.append(a_x)
        if 'v' in self.modal:
            h.append(v_x)
        h = torch.cat(h, dim=1)
        y = self.fc(h)
        # h = F.relu(self.fc1(h))
        # h =self.dropout(h)
        # y = self.fc2(h)

        return y
