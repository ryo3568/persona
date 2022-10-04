import torch 
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence 
import pickle 
import pandas as pd 
import numpy as np 
from utils.Standardizing import Standardizing

class Hazumi1911Dataset(Dataset):

    def __init__(self, test_file, train=True, scaler=None):
    
        path = '../data/Hazumi1911/Hazumi1911_features/Hazumi1911_features_persona.pkl'


        self.videoIDs, self.videoLabels, self.videoText, self.videoAudio,\
        self.videoVisual, self.videoSentence, self.Vid = pickle.load(open(path, 'rb'), encoding='utf-8')

        self.keys = [] 

        if train:
            for x in self.Vid:
                if x != test_file:
                    self.keys.append(x) 
            self.scaler_text = Standardizing()
            self.scaler_audio = Standardizing()
            self.scaler_visual = Standardizing()
            self.scaler_text.fit(self.videoText, self.keys)
            self.scaler_audio.fit(self.videoAudio, self.keys)
            self.scaler_visual.fit(self.videoVisual, self.keys)
            self.scaler = (self.scaler_text, self.scaler_audio, self.scaler_visual)
        else:
            self.keys.append(test_file)
            self.scaler_text, self.scaler_audio, self.scaler_visual = scaler 

        self.len = len(self.keys) 

        
    def __getitem__(self, index):
        vid = self.keys[index] 
        return torch.FloatTensor(self.scaler_text.transform(self.videoText[vid])),\
               torch.FloatTensor(self.scaler_visual.transform(self.videoVisual[vid])),\
               torch.FloatTensor(self.scaler_audio.transform(self.videoAudio[vid])),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.FloatTensor(self.videoLabels[vid]),\
               vid
        

    def __len__(self):
        return self.len 

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]


class Hazumi1911SentimentDataset(Dataset):

    def __init__(self, test_file, train=True, scaler=None):
    
        path = '../data/Hazumi1911/Hazumi1911_features/Hazumi1911_features_sentiment.pkl'


        self.videoIDs, self.videoLabels, self.videoText, self.videoAudio,\
        self.videoVisual, self.videoSentence, self.Vid = pickle.load(open(path, 'rb'), encoding='utf-8')

        self.keys = [] 

        if train:
            for x in self.Vid:
                if x != test_file:
                    self.keys.append(x) 
            self.scaler_text = Standardizing()
            self.scaler_audio = Standardizing()
            self.scaler_visual = Standardizing()
            self.scaler_text.fit(self.videoText, self.keys)
            self.scaler_audio.fit(self.videoAudio, self.keys)
            self.scaler_visual.fit(self.videoVisual, self.keys)
            self.scaler = (self.scaler_text, self.scaler_audio, self.scaler_visual)
        else:
            self.keys.append(test_file)
            self.scaler_text, self.scaler_audio, self.scaler_visual = scaler

        self.len = len(self.keys) 

        
    def __getitem__(self, index):
        vid = self.keys[index] 
        return torch.FloatTensor(self.scaler_text.transform(self.videoText[vid])),\
               torch.FloatTensor(self.scaler_visual.transform(self.videoVisual[vid])),\
               torch.FloatTensor(self.scaler_audio.transform(self.videoAudio[vid])),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len 

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]