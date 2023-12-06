import glob
import pickle 
import numpy as np
from sklearn import preprocessing
import torch 

from model import UnimodalFNN
from utils import profiling

class HazumiDataset_torch(torch.utils.data.Dataset):
    def __init__(self, version, ids, sscaler=None, modal='t', ss=False):
        path = f'../data/Hazumi_features/Hazumi{version}_features.pkl'
        _, SS_binary, SS_ternary, _, TS_binary, TS_ternary, _, _, Text, Audio, Visual, _ =\
            pickle.load(open(path, 'rb'), encoding='utf-8')

        if modal == 't':
            data = Text 
        elif modal == 'a':
            data = Audio
        elif modal == 'v':
            data = Visual
        
        if ss:
            label = SS_ternary
        else:
            label = TS_ternary

        X, y = [], []
        for id in ids:
            X.extend(data[id])
            y.extend(label[id])

        if modal != 't':
            if sscaler is None:
                self.sscaler = preprocessing.StandardScaler()
                X = self.sscaler.fit_transform(X)
            else:
                self.sscaler = sscaler
                X = self.sscaler.transform(X)
        else:
            self.sscaler = sscaler
        
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def get_sscaler(self):
        return self.sscaler
    
    def __getitem__(self, idx):
        features_x = torch.FloatTensor(self.X[idx])
        labels = torch.LongTensor([self.y[idx]])
        return features_x, labels

class HazumiTestDataset_torch(torch.utils.data.Dataset):
    def __init__(self, version, id, sscaler=None, modal='t', ss=False):
        path = f'../data/Hazumi_features/Hazumi{version}_features.pkl'
        _, SS_binary, SS_ternary, _, TS_binary, TS_ternary, _, _, Text, Audio, Visual, _ =\
            pickle.load(open(path, 'rb'), encoding='utf-8')

        if modal == 't':
            data = Text 
        elif modal == 'a':
            data = Audio
        elif modal == 'v':
            data = Visual

        if ss:
            label = SS_ternary
        else:
            label = TS_ternary

        X, y = [], []
        X.extend(data[id])
        y.extend(label[id])

        if modal != 't':
            X = sscaler.transform(X)
        
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        features_x = torch.FloatTensor(self.X[idx])
        labels = torch.LongTensor([self.y[idx]])
        return features_x, labels

class HazumiDataset_multi(torch.utils.data.Dataset):
    def __init__(self, version, ids, a_scaler=None, v_scaler=None, ss=False):
        path = f'../data/Hazumi_features/Hazumi{version}_features.pkl'
        _, SS_binary, SS_ternary, _, TS_binary, TS_ternary, _, _, Text, Audio, Visual, _ =\
            pickle.load(open(path, 'rb'), encoding='utf-8')

        # self.a_scaler = a_scaler
        # self.v_scaler = v_scaler

        if ss:
            label = SS_ternary
        else:
            label = TS_ternary

        t_X, a_X, v_X, y = [], [], [], []
        for id in ids:
            t_X.extend(Text[id])
            a_X.extend(Audio[id])
            v_X.extend(Visual[id])
            y.extend(label[id])

        if a_scaler is None:
            self.a_scaler = preprocessing.StandardScaler()
            self.v_scaler = preprocessing.StandardScaler()
            a_X = self.a_scaler.fit_transform(a_X)
            v_X = self.v_scaler.fit_transform(v_X)
        else:
            self.a_scaler = a_scaler
            self.v_scaler = v_scaler
            a_X = self.a_scaler.fit_transform(a_X)
            v_X = self.v_scaler.fit_transform(v_X)
        
        self.t_X = t_X
        self.a_X = a_X
        self.v_X = v_X
        self.y = y

    def __len__(self):
        return len(self.t_X)
    
    def get_sscaler(self):
        return self.a_scaler, self.v_scaler
    
    def __getitem__(self, idx):
        t_x = torch.FloatTensor(self.t_X[idx])
        a_x = torch.FloatTensor(self.a_X[idx])
        v_x = torch.FloatTensor(self.v_X[idx])
        labels = torch.LongTensor([self.y[idx]])
        return t_x, a_x, v_x, labels

class HazumiTestDataset_multi(torch.utils.data.Dataset):
    def __init__(self, version, id, a_scaler=None, v_scaler=None, ss=False):
        path = f'../data/Hazumi_features/Hazumi{version}_features.pkl'
        _, SS_binary, SS_ternary, _, TS_binary, TS_ternary, _, _, Text, Audio, Visual, _ =\
            pickle.load(open(path, 'rb'), encoding='utf-8')

        if ss:
            label = SS_ternary
        else:
            label = TS_ternary

        t_X, a_X, v_X, y = [], [], [], []
        t_X.extend(Text[id])
        a_X.extend(Audio[id])
        v_X.extend(Visual[id])
        y.extend(label[id])

        a_X = a_scaler.transform(a_X)
        v_X = v_scaler.transform(v_X)
        
        self.t_X = t_X
        self.a_X = a_X
        self.v_X = v_X
        self.y = y

    def __len__(self):
        return len(self.t_X)
    
    def __getitem__(self, idx):
        t_x = torch.FloatTensor(self.t_X[idx])
        a_x = torch.FloatTensor(self.a_X[idx])
        v_x = torch.FloatTensor(self.v_X[idx])
        labels = torch.LongTensor([self.y[idx]])
        return t_x, a_x, v_x, labels

class HazumiDataset_better(torch.utils.data.Dataset):
    def __init__(self, version, ids, a_scaler=None, v_scaler=None, ss=False):
        path = f'../data/Hazumi_features/Hazumi{version}_features.pkl'
        _, SS_binary, SS_ternary, _, TS_binary, TS_ternary, _, _, Text, Audio, Visual, _ =\
            pickle.load(open(path, 'rb'), encoding='utf-8')

        # self.a_scaler = a_scaler
        # self.v_scaler = v_scaler

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        t_model = UnimodalFNN(input_dim=768, num_classes=3).to(device)
        t_file = glob.glob(f"model/ts-{0}-{0}-t-*")[0]
        t_model.load_state_dict(torch.load(t_file))
        t_model.fc5 = torch.nn.Identity()


        a_f_model = UnimodalFNN(input_dim=384, num_classes=3).to(device)
        a_f_file = glob.glob(f"model/ts-{1}-{0}-a-*")[0]
        a_f_model.load_state_dict(torch.load(a_f_file))
        a_f_model.fc5 = torch.nn.Identity()

        a_m_model = UnimodalFNN(input_dim=384, num_classes=3).to(device)
        a_m_file = glob.glob(f"model/ts-{1}-{1}-a-*")[0]
        a_m_model.load_state_dict(torch.load(a_m_file))
        a_m_model.fc5 = torch.nn.Identity()


        v_u_model = UnimodalFNN(input_dim=66, num_classes=3).to(device)
        v_u_file = glob.glob(f"model/ts-{2}-{0}-v-*")[0]
        v_u_model.load_state_dict(torch.load(v_u_file))
        v_u_model.fc5 = torch.nn.Identity()

        v_o_model = UnimodalFNN(input_dim=66, num_classes=3).to(device)
        v_o_file = glob.glob(f"model/ts-{2}-{1}-v-*")[0]
        v_o_model.load_state_dict(torch.load(v_o_file))
        v_o_model.fc5 = torch.nn.Identity()

        t_model.eval() 
        a_f_model.eval() 
        a_m_model.eval() 
        v_u_model.eval()
        v_o_model.eval()

        if ss:
            label = SS_ternary
        else:
            label = TS_ternary

        t_X, a_X, v_X, y = [], [], [], []
        for id in ids:
            _t = torch.tensor(Text[id]).to(device)
            _t = t_model(_t).tolist()
            t_X.extend(_t)

            if profiling(1, id) == 0:
                _a = torch.tensor(Audio[id]).to(device)
                _a = a_f_model(_a).tolist()
            else:
                _a = torch.tensor(Audio[id]).to(device)
                _a = a_m_model(_a).tolist()
            a_X.extend(_a)

            if profiling(2, id) == 0:
                _v = torch.tensor(Visual[id]).to(device)
                _v = v_u_model(_v).tolist()
            else:
                _v = torch.tensor(Visual[id]).to(device)
                _v = v_o_model(_v).tolist()
            v_X.extend(_v)

            y.extend(label[id])

        if a_scaler is None:
            self.a_scaler = preprocessing.StandardScaler()
            self.v_scaler = preprocessing.StandardScaler()
            a_X = self.a_scaler.fit_transform(a_X)
            v_X = self.v_scaler.fit_transform(v_X)
        else:
            self.a_scaler = a_scaler
            self.v_scaler = v_scaler
            a_X = self.a_scaler.fit_transform(a_X)
            v_X = self.v_scaler.fit_transform(v_X)
        
        self.t_X = t_X
        self.a_X = a_X
        self.v_X = v_X
        self.y = y

    def __len__(self):
        return len(self.t_X)
    
    def get_sscaler(self):
        return self.a_scaler, self.v_scaler
    
    def __getitem__(self, idx):
        t_x = torch.FloatTensor(self.t_X[idx])
        a_x = torch.FloatTensor(self.a_X[idx])
        v_x = torch.FloatTensor(self.v_X[idx])
        labels = torch.LongTensor([self.y[idx]])
        return t_x, a_x, v_x, labels

class HazumiTestDataset_better(torch.utils.data.Dataset):
    def __init__(self, version, id, a_scaler=None, v_scaler=None, ss=False):
        path = f'../data/Hazumi_features/Hazumi{version}_features.pkl'
        _, SS_binary, SS_ternary, _, TS_binary, TS_ternary, _, _, Text, Audio, Visual, _ =\
            pickle.load(open(path, 'rb'), encoding='utf-8')
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        t_model = UnimodalFNN(input_dim=768, num_classes=3).to(device)
        # t_file = glob.glob(f"model/{'ss' if args.ss else 'ts'}-{_t[0]}-{_t[1]}-t-*")[0]
        t_file = glob.glob(f"model/ts-{0}-{0}-t-*")[0]
        t_model.load_state_dict(torch.load(t_file))
        t_model.fc5 = torch.nn.Identity()

        a_f_model = UnimodalFNN(input_dim=384, num_classes=3).to(device)
        a_m_model = UnimodalFNN(input_dim=384, num_classes=3).to(device)
        a_f_file = glob.glob(f"model/ts-{1}-{0}-a-*")[0]
        a_m_file = glob.glob(f"model/ts-{1}-{1}-a-*")[0]
        a_f_model.load_state_dict(torch.load(a_f_file))
        a_m_model.load_state_dict(torch.load(a_m_file))
        a_f_model.fc5 = torch.nn.Identity()
        a_m_model.fc5 = torch.nn.Identity()

        v_u_model = UnimodalFNN(input_dim=66, num_classes=3).to(device)
        v_o_model = UnimodalFNN(input_dim=66, num_classes=3).to(device)
        v_u_file = glob.glob(f"model/ts-{2}-{0}-v-*")[0]
        v_o_file = glob.glob(f"model/ts-{2}-{1}-v-*")[0]
        v_u_model.load_state_dict(torch.load(v_u_file))
        v_o_model.load_state_dict(torch.load(v_o_file))
        v_u_model.fc5 = torch.nn.Identity()
        v_o_model.fc5 = torch.nn.Identity()

        t_model.eval() 
        a_f_model.eval() 
        a_m_model.eval() 
        v_u_model.eval()
        v_o_model.eval()

        if ss:
            label = SS_ternary
        else:
            label = TS_ternary

        t_X, a_X, v_X, y = [], [], [], []

        _t = torch.tensor(Text[id]).to(device)
        _t = t_model(_t).tolist()
        t_X.extend(_t)

        if profiling(1, id) == 0:
            _a = torch.tensor(Audio[id]).to(device)
            _a = a_f_model(_a).tolist()
        else:
            _a = torch.tensor(Audio[id]).to(device)
            _a = a_m_model(_a).tolist()
        a_X.extend(_a)

        if profiling(2, id) == 0:
            _v = torch.tensor(Visual[id]).to(device)
            _v = v_u_model(_v).tolist()
        else:
            _v = torch.tensor(Visual[id]).to(device)
            _v = v_o_model(_v).tolist()
        v_X.extend(_v)

        y.extend(label[id])
            
        a_X = a_scaler.transform(a_X)
        v_X = v_scaler.transform(v_X)
        
        self.t_X = t_X
        self.a_X = a_X
        self.v_X = v_X
        self.y = y

    def __len__(self):
        return len(self.t_X)
    
    def __getitem__(self, idx):
        t_x = torch.FloatTensor(self.t_X[idx])
        a_x = torch.FloatTensor(self.a_X[idx])
        v_x = torch.FloatTensor(self.v_X[idx])
        labels = torch.LongTensor([self.y[idx]])
        return t_x, a_x, v_x, labels
