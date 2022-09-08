import torch 
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence 
import pickle 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 

class Hazumi1911Dataset(Dataset):

    def __init__(self, test_file, train=True, scaler=None):
    
        path = '../data/Hazumi1911/Hazumi1911_features/Hazumi1911_features.pkl'


        self.videoIDs, self.videoLabels, self.videoText, self.videoAudio,\
        self.videoVisual, self.videoSentence, self.Vid = pickle.load(open(path, 'rb'), encoding='utf-8')

        self.keys = [] 

        if train:
            for x in self.Vid:
                if x != test_file:
                    self.keys.append(x) 
            # self.scaler = standardize()
            self.scaler = scaler
        else:
            self.keys.append(test_file)
            self.scaler = scaler 

        self.len = len(self.keys) 

        
    def __getitem__(self, index):
        vid = self.keys[index] 
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.FloatTensor([self.videoLabels[vid][4]]),\
               vid

    def __len__(self):
        return self.len 

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]