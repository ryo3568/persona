import pickle 
import numpy as np
from sklearn import preprocessing
import torch 

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

class HazumiDataset_multiv2(torch.utils.data.Dataset):
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
        IDs = []
        for id in ids:
            t_X.extend(Text[id])
            a_X.extend(Audio[id])
            v_X.extend(Visual[id])
            y.extend(label[id])
            IDs.extend([id] * len(label[id]))
        
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
        self.IDs = IDs

    def __len__(self):
        return len(self.t_X)
    
    def get_sscaler(self):
        return self.a_scaler, self.v_scaler
    
    def __getitem__(self, idx):
        t_x = torch.FloatTensor(self.t_X[idx])
        a_x = torch.FloatTensor(self.a_X[idx])
        v_x = torch.FloatTensor(self.v_X[idx])
        labels = torch.LongTensor([self.y[idx]])
        id = self.IDs[idx]
        return t_x, a_x, v_x, labels, id

class HazumiTestDataset_multiv2(torch.utils.data.Dataset):
    def __init__(self, version, id, a_scaler=None, v_scaler=None, ss=False):
        path = f'../data/Hazumi_features/Hazumi{version}_features.pkl'
        _, SS_binary, SS_ternary, _, TS_binary, TS_ternary, _, _, Text, Audio, Visual, _ =\
            pickle.load(open(path, 'rb'), encoding='utf-8')

        if ss:
            label = SS_ternary
        else:
            label = TS_ternary

        t_X, a_X, v_X, y = [], [], [], []
        IDs = []
        t_X.extend(Text[id])
        a_X.extend(Audio[id])
        v_X.extend(Visual[id])
        y.extend(label[id])
        IDs.extend([id]*len(label[id]))

        a_X = a_scaler.transform(a_X)
        v_X = v_scaler.transform(v_X)
        
        self.t_X = t_X
        self.a_X = a_X
        self.v_X = v_X
        self.y = y
        self.IDs = IDs

    def __len__(self):
        return len(self.t_X)
    
    def __getitem__(self, idx):
        t_x = torch.FloatTensor(self.t_X[idx])
        a_x = torch.FloatTensor(self.a_X[idx])
        v_x = torch.FloatTensor(self.v_X[idx])
        labels = torch.LongTensor([self.y[idx]])
        id = self.IDs[idx]
        return t_x, a_x, v_x, labels, id
