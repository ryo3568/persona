import torch 
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence 
import pickle 
import pandas as pd 
from utils.Standardizing import Standardizing


class HazumiDataset(Dataset):

    def __init__(self, version, test_file, train=True, scaler=None):
    
        path = f'../data/Hazumi_features/Hazumi{version}_features.pkl'

        self.SS, self.TS, self.SP, self.TP, \
        self.text, self.audio, self.visual, self.bio, self.vid = pickle.load(open(path, 'rb'), encoding='utf-8')

        self.keys = [] 

        if train:
            for x in self.vid:
                if x != test_file:
                    self.keys.append(x) 
            self.scaler_audio = Standardizing()
            self.scaler_visual = Standardizing()
            self.scaler_bio = Standardizing()
            self.scaler_audio.fit(self.audio, self.keys)
            self.scaler_visual.fit(self.visual, self.keys)
            self.scaler_bio.fit(self.bio, self.keys)
            self.scaler = (self.scaler_audio, self.scaler_visual, self.scaler_bio)
        else:
            self.keys.append(test_file)
            self.scaler_audio, self.scaler_visual, self.scaler_bio = scaler 

        self.len = len(self.keys) 

        
    def __getitem__(self, index):
        vid = self.keys[index] 
        return torch.FloatTensor(self.text[vid]),\
            torch.FloatTensor(self.scaler_visual.transform(self.visual[vid])),\
            torch.FloatTensor(self.scaler_audio.transform(self.audio[vid])),\
            torch.FloatTensor(self.scaler_bio.transform(self.bio[vid])),\
            torch.FloatTensor(self.TP[vid]),\
            torch.LongTensor(self.SS[vid]),\
            vid

    def __len__(self):
        return self.len 

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


# class HazumiDataset(Dataset):

#     def __init__(self, test_file, train=True, scaler=None):
    
#         path = '../data/Hazumi_features/Hazumi1911_features_self.pkl'

#         self.SS, self.SS_ternary, self.SP, self.SP_binary, self.SP_cluster, \
#         self.text, self.audio, self.visual, self.vid = pickle.load(open(path, 'rb'), encoding='utf-8')

#         self.keys = [] 

#         if train:
#             for x in self.vid:
#                 if x != test_file:
#                     self.keys.append(x) 
#             self.scaler_audio = Standardizing()
#             self.scaler_visual = Standardizing()
#             self.scaler_audio.fit(self.audio, self.keys)
#             self.scaler_visual.fit(self.visual, self.keys)
#             self.scaler = (self.scaler_audio, self.scaler_visual)
#         else:
#             self.keys.append(test_file)
#             self.scaler_audio, self.scaler_visual = scaler 

#         self.len = len(self.keys) 

        
#     def __getitem__(self, index):
#         vid = self.keys[index] 
#         return torch.FloatTensor(self.text[vid]),\
#             torch.FloatTensor(self.scaler_visual.transform(self.visual[vid])),\
#             torch.FloatTensor(self.scaler_audio.transform(self.audio[vid])),\
#             torch.LongTensor([self.SP_cluster[vid]]),\
#             torch.LongTensor(self.SS_ternary[vid]),\
#             vid

#     def __len__(self):
#         return self.len 

#     def collate_fn(self, data):
#         dat = pd.DataFrame(data)

#         return [pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]


class HazumiDataset_sweep(Dataset):
    """マルチタスク学習用データセット
 
    Sweep用のデータセット

    """

    def __init__(self):
    
        path = '../data/Hazumi_features/Hazumi1911_features.pkl'

        self.TS, self.TS_ternary, self.TP, self.TP_binary, self.TP_cluster, \
        self.text, self.audio, self.visual, self.vid = pickle.load(open(path, 'rb'), encoding='utf-8')

        self.keys = [] 

        for x in self.vid:
            self.keys.append(x) 
        self.scaler_audio = Standardizing()
        self.scaler_visual = Standardizing()
        self.scaler_audio.fit(self.audio, self.keys)
        self.scaler_visual.fit(self.visual, self.keys)
        self.scaler = (self.scaler_audio, self.scaler_visual)

        self.len = len(self.keys) 

        
    def __getitem__(self, index):
        vid = self.keys[index] 

        return torch.FloatTensor(self.text[vid]),\
            torch.FloatTensor(self.scaler_visual.transform(self.visual[vid])),\
            torch.FloatTensor(self.scaler_audio.transform(self.audio[vid])),\
            torch.FloatTensor(self.TP_binary[vid]),\
            torch.LongTensor(self.TS_ternary[vid]),\
            vid

    def __len__(self):
        return self.len 

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]











