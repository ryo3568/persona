import torch 
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence 
import pickle 
import pandas as pd 
import utils
from utils.Standardizing import Standardizing

class HazumiDataset(torch.utils.data.Dataset):
    def __init__(self, test_file="", train=True):
        super().__init__()

        path = f'../data/Hazumi_features/Hazumi1911_features.pkl'

        self.SS, self.TS, self.SP, self.TP, \
        self.text, self.audio, self.visual, self.vid = pickle.load(open(path, 'rb'), encoding='utf-8')

        self.text_list = [] 
        self.ss_list = []
        self.ts_list = []

        if train:
            for x in self.vid:
                if x != test_file:
                    self.text_list.extend(self.text[x]) 
                    self.ss_list.extend(self.SS[x])
                    self.ts_list.extend(self.TS[x])

        else:
            self.text_list.extend(self.text[test_file])
            self.ss_list.extend(self.SS[test_file])
            self.ts_list.extend(self.TS[test_file])
        

    def __len__(self):
        return len(self.text_list)
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.text_list[index]), \
            self.ss_list[index], \
            self.ts_list[index], \

class HazumiDatasetforRNN(Dataset):
    def __init__(self, version, test_file, train=True, scaler=None):
    
        path = f'../data/Hazumi_features/Hazumi{version}_features.pkl'

        self.SS, self.TS, self.SP, self.TP, \
        self.text, self.audio, self.visual, self.rawtext, self.vid = pickle.load(open(path, 'rb'), encoding='utf-8')

        self.keys = [] 

        if train:
            for x in self.vid:
                if x != test_file:
                    self.keys.append(x) 
            self.scaler_audio = Standardizing()
            self.scaler_visual = Standardizing()
            self.scaler_audio.fit(self.audio, self.keys)
            self.scaler_visual.fit(self.visual, self.keys)
            self.scaler = (self.scaler_audio, self.scaler_visual)
        else:
            self.keys.append(test_file)
            self.scaler_audio, self.scaler_visual = scaler 

        self.len = len(self.keys) 

        self.TP = utils.Normalization(self.TP)

        
    def __getitem__(self, index):
        vid = self.keys[index] 
        return torch.FloatTensor(self.text[vid]),\
            torch.FloatTensor(self.scaler_visual.transform(self.visual[vid])),\
            torch.FloatTensor(self.scaler_audio.transform(self.audio[vid])),\
            torch.FloatTensor(self.TP[vid]),\
            torch.LongTensor(self.SS[vid]),\
            torch.LongTensor(self.TS[vid]),\
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


class HazumiDataset_sweep(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        path = f'../data/Hazumi_features/Hazumi1911_features.pkl'

        self.SS, self.TS, self.SP, self.TP, \
        self.text, self.audio, self.visual, self.vid = pickle.load(open(path, 'rb'), encoding='utf-8')

        self.text_list = [] 
        self.ss_list = []
        self.ts_list = []

        for id in self.vid:
            self.text_list.extend(self.text[id]) 
            self.ss_list.extend(self.SS[id])
            self.ts_list.extend(self.TS[id])

    def __len__(self):
        return len(self.text_list)
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.text_list[index]), \
            self.ss_list[index], \
            self.ts_list[index], \

# class HazumiDataset_sweep(Dataset):
#     """マルチタスク学習用データセット
 
#     Sweep用のデータセット

#     """

#     def __init__(self):
    
#         path = '../data/Hazumi_features/Hazumi1911_features.pkl'

#         self.SS, self.TS, self.SP, self.TP, self.text, self.audio, self.visual, self.vid \
#             = pickle.load(open(path, 'rb'), encoding='utf-8')

#         self.text_list = [] 
#         self.ss_list = [] 
#         self.ts_list = []

#         for x in self.vid:
#             self.keys.append(x) 
#         self.scaler_audio = Standardizing()
#         self.scaler_visual = Standardizing()
#         self.scaler_audio.fit(self.audio, self.keys)
#         self.scaler_visual.fit(self.visual, self.keys)
#         self.scaler = (self.scaler_audio, self.scaler_visual)

#         self.len = len(self.keys) 

        
#     def __getitem__(self, index):
#         vid = self.keys[index] 

#         return torch.FloatTensor(self.text[vid]),\
#             torch.FloatTensor(self.scaler_visual.transform(self.visual[vid])),\
#             torch.FloatTensor(self.scaler_audio.transform(self.audio[vid])),\
#             torch.FloatTensor(self.TP_binary[vid]),\
#             torch.LongTensor(self.TS_ternary[vid]),\
#             vid

#     def __len__(self):
#         return self.len 

#     def collate_fn(self, data):
#         dat = pd.DataFrame(data)
#         return [pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]











