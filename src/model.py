import glob
import torch
import torch.nn as nn 
import torch.nn.functional as F 

class UnimodalFNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(UnimodalFNN, self).__init__() 
        # self.fc1 = nn.Linear(input_dim, 64)
        # self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64, 32)
        # self.fc4 = nn.Linear(32, 32)
        # self.fc5 = nn.Linear(32, num_classes)

        # self.fc1 = nn.Linear(input_dim, int(input_dim / 2))
        # self.fc2 = nn.Linear(int(input_dim / 2), int(input_dim / 2))
        # self.fc3 = nn.Linear(int(input_dim / 2), int(input_dim / 4))
        # self.fc4 = nn.Linear(int(input_dim / 4), 32)
        # self.fc5 = nn.Linear(16, num_classes)

        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, int(input_dim / 2))
        self.fc3 = nn.Linear(int(input_dim / 2), int(input_dim / 2))
        self.fc4 = nn.Linear(int(input_dim / 2), 32)
        self.fc5 = nn.Linear(32, num_classes)

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
    def __init__(self, input_dim, num_classes, ss, pmode, pgroup, modal):
        super(LatefusionFNN, self).__init__() 
        self.t_model = UnimodalFNN(input_dim=768, num_classes=num_classes)
        t_file = glob.glob(f"model/{'ss' if ss else 'ts'}-{pmode}-{pgroup}-t-*")[0]
        # t_file = glob.glob(f"model/{'ss' if ss else 'ts'}-0-0-t-*")[0]
        self.t_model.load_state_dict(torch.load(t_file))
        self.t_model.fc5 = torch.nn.Identity()
        for param in self.t_model.parameters():
            param.requires_grad = False


        self.a_model = UnimodalFNN(input_dim=384, num_classes=num_classes)
        a_file = glob.glob(f"model/{'ss' if ss else 'ts'}-{pmode}-{pgroup}-a-*")[0]
        # a_file = glob.glob(f"model/{'ss' if ss else 'ts'}-0-0-a-*")[0]
        self.a_model.load_state_dict(torch.load(a_file))
        self.a_model.fc5 = torch.nn.Identity()
        for param in self.a_model.parameters():
            param.requires_grad = False


        self.v_model = UnimodalFNN(input_dim=66, num_classes=num_classes)
        v_file = glob.glob(f"model/{'ss' if ss else 'ts'}-{pmode}-{pgroup}-v-*")[0]
        # v_file = glob.glob(f"model/{'ss' if ss else 'ts'}-0-0-v-*")[0]
        self.v_model.load_state_dict(torch.load(v_file))
        self.v_model.fc5 = torch.nn.Identity()
        for param in self.v_model.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(input_dim, num_classes)
        self.fc1 = nn.Linear(input_dim, int(input_dim / 2))
        self.fc2 = nn.Linear(int(input_dim/2), num_classes)
        self.dropout = nn.Dropout(0.3)

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
        # y = self.fc(h)
        h = F.relu(self.fc1(h))
        h =self.dropout(h)
        y = self.fc2(h)

        return y

class LatefusionFNNv2(nn.Module):
    def __init__(self, input_dim, num_classes, ss, pmode, pgroup, modal):
        super(LatefusionFNNv2, self).__init__() 
        self.t_model = UnimodalFNN(input_dim=768, num_classes=num_classes)
        # t_file = glob.glob(f"model/{'ss' if ss else 'ts'}-{pmode}-{pgroup}-t-*")[0]
        t_file = glob.glob(f"model/{'ss' if ss else 'ts'}-0-0-t-*")[0]
        self.t_model.load_state_dict(torch.load(t_file))
        self.t_model.fc5 = torch.nn.Identity()
        # for param in self.t_model.parameters():
        #     param.requires_grad = False


        self.f_model = UnimodalFNN(input_dim=384, num_classes=num_classes)
        # a_file = glob.glob(f"model/{'ss' if ss else 'ts'}-{pmode}-{pgroup}-a-*")[0]
        a_file = glob.glob(f"model/{'ss' if ss else 'ts'}-1-0-a-*")[0]
        self.f_model.load_state_dict(torch.load(a_file))
        self.f_model.fc5 = torch.nn.Identity()
        # for param in self.a_model.parameters():
        #     param.requires_grad = False

        self.m_model = UnimodalFNN(input_dim=384, num_classes=num_classes)
        # a_file = glob.glob(f"model/{'ss' if ss else 'ts'}-{pmode}-{pgroup}-a-*")[0]
        a_file = glob.glob(f"model/{'ss' if ss else 'ts'}-1-1-a-*")[0]
        self.m_model.load_state_dict(torch.load(a_file))
        self.m_model.fc5 = torch.nn.Identity()
        # for param in self.a_model.parameters():
        #     param.requires_grad = False


        self.v_model = UnimodalFNN(input_dim=66, num_classes=num_classes)
        # v_file = glob.glob(f"model/{'ss' if ss else 'ts'}-{pmode}-{pgroup}-v-*")[0]
        v_file = glob.glob(f"model/{'ss' if ss else 'ts'}-0-0-v-*")[0]
        self.v_model.load_state_dict(torch.load(v_file))
        self.v_model.fc5 = torch.nn.Identity()
        # for param in self.v_model.parameters():
        #     param.requires_grad = False
        self.fc = nn.Linear(input_dim, num_classes)
        self.fc1 = nn.Linear(input_dim, int(input_dim / 2))
        self.fc2 = nn.Linear(int(input_dim/2), num_classes)
        self.dropout = nn.Dropout(0.3)

        self.modal = modal
    
    def forward(self, t_x, a_x, v_x, id):
        h = []
        id = id[0]
        if 't' in self.modal:
            t_y = self.t_model(t_x)
            h.append(t_y)
        if 'a' in self.modal:
            if id[4] == 'F':
                a_y = self.f_model(a_x)
                h.append(a_y)
            else:
                a_y = self.m_model(a_x)
                h.append(a_y)
        if 'v' in self.modal:
            v_y = self.v_model(v_x)
            h.append(v_y)
        h = torch.cat(h, dim=1)
        # y = self.fc(h)
        h = F.relu(self.fc1(h))
        h =self.dropout(h)
        y = self.fc2(h)

        return y