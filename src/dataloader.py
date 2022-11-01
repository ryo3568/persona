import torch 
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence 
import pickle 
import pandas as pd 
import numpy as np 
from utils.Standardizing import Standardizing

class HazumiDataset(Dataset):
    """タイトル

    説明文

    """


    def __init__(self, test_file, target=5, train=True, scaler=None):
    
        path = '../data/Hazumi_features/Hazumi1911_features.pkl'


        self.SS_ternary, self.TS_ternary, self.sentiment, self.third_sentiment, self.persona, self.third_persona,\
        self.text, self.audio, self.visual, self.vid = pickle.load(open(path, 'rb'), encoding='utf-8')

        self.keys = [] 
        self.target = target

        if train:
            for x in self.vid:
                if x != test_file:
                    self.keys.append(x) 
            self.scaler_text = Standardizing()
            self.scaler_audio = Standardizing()
            self.scaler_visual = Standardizing()
            self.scaler_text.fit(self.text, self.keys)
            self.scaler_audio.fit(self.audio, self.keys)
            self.scaler_visual.fit(self.visual, self.keys)
            self.scaler = (self.scaler_text, self.scaler_audio, self.scaler_visual)
        else:
            self.keys.append(test_file)
            self.scaler_text, self.scaler_audio, self.scaler_visual = scaler 

        self.len = len(self.keys) 

        
    def __getitem__(self, index):
        vid = self.keys[index] 

        if self.target == 5:
            return torch.FloatTensor(self.scaler_text.transform(self.text[vid])),\
                torch.FloatTensor(self.scaler_visual.transform(self.visual[vid])),\
                torch.FloatTensor(self.scaler_audio.transform(self.audio[vid])),\
                torch.FloatTensor(self.third_persona[vid]),\
                vid

        return torch.FloatTensor(self.scaler_text.transform(self.text[vid])),\
            torch.FloatTensor(self.scaler_visual.transform(self.visual[vid])),\
            torch.FloatTensor(self.scaler_audio.transform(self.audio[vid])),\
            torch.FloatTensor([self.third_persona[vid][self.target]]),\
            vid
        

    def __len__(self):
        return self.len 

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]


class HazumiMultiTaskDataset(Dataset):
    """マルチタスク学習用データセット

    説明文

    """

    def __init__(self, test_file, target=5, train=True, scaler=None):
    
        path = '../data/Hazumi_features/Hazumi1911_features.pkl'


        self.SS_ternary, self.TS_ternary, self.sentiment, self.third_sentiment, self.persona, self.third_persona,\
        self.text, self.audio, self.visual, self.vid = pickle.load(open(path, 'rb'), encoding='utf-8')

        self.keys = [] 
        self.target = target

        if train:
            for x in self.vid:
                if x != test_file:
                    self.keys.append(x) 
            self.scaler_text = Standardizing()
            self.scaler_audio = Standardizing()
            self.scaler_visual = Standardizing()
            self.scaler_text.fit(self.text, self.keys)
            self.scaler_audio.fit(self.audio, self.keys)
            self.scaler_visual.fit(self.visual, self.keys)
            self.scaler = (self.scaler_text, self.scaler_audio, self.scaler_visual)
        else:
            self.keys.append(test_file)
            self.scaler_text, self.scaler_audio, self.scaler_visual = scaler 

        self.len = len(self.keys) 

        
    def __getitem__(self, index):
        vid = self.keys[index] 

        if self.target == 5:
            return torch.FloatTensor(self.scaler_text.transform(self.text[vid])),\
                torch.FloatTensor(self.scaler_visual.transform(self.visual[vid])),\
                torch.FloatTensor(self.scaler_audio.transform(self.audio[vid])),\
                torch.FloatTensor(self.third_persona[vid]),\
                torch.FloatTensor(self.third_sentiment[vid]),\
                vid

        return torch.FloatTensor(self.scaler_text.transform(self.text[vid])),\
            torch.FloatTensor(self.scaler_visual.transform(self.visual[vid])),\
            torch.FloatTensor(self.scaler_audio.transform(self.audio[vid])),\
            torch.FloatTensor([self.third_persona[vid][self.target]]),\
            vid
        

    def __len__(self):
        return self.len 

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]



class HazumiSentimentDataset(Dataset):
    """心象推定による事前学習用データセット

    説明文

    """

    def __init__(self, test_file, train=True, scaler=None):
    
        path = '../data/Hazumi_features/Hazumi1911_features.pkl'


        self.SS_ternary, self.TS_ternary, self.sentiment, self.third_sentiment, self.persona, self.third_persona,\
        self.text, self.audio, self.visual, self.vid = pickle.load(open(path, 'rb'), encoding='utf-8')

        self.keys = [] 

        if train:
            for x in self.vid:
                if x != test_file:
                    self.keys.append(x) 
            self.scaler_text = Standardizing()
            self.scaler_audio = Standardizing()
            self.scaler_visual = Standardizing()
            self.scaler_text.fit(self.text, self.keys)
            self.scaler_audio.fit(self.audio, self.keys)
            self.scaler_visual.fit(self.visual, self.keys)
            self.scaler = (self.scaler_text, self.scaler_audio, self.scaler_visual)
        else:
            self.keys.append(test_file)
            self.scaler_text, self.scaler_audio, self.scaler_visual = scaler

        self.len = len(self.keys) 

        
    def __getitem__(self, index):
        vid = self.keys[index] 
        return torch.FloatTensor(self.text[vid]),\
               torch.FloatTensor(self.scaler_visual.transform(self.visual[vid])),\
               torch.FloatTensor(self.scaler_audio.transform(self.audio[vid])),\
               torch.FloatTensor([1]*len(self.third_sentiment[vid])),\
               torch.FloatTensor(self.third_persona[vid]),\
               torch.LongTensor(self.third_sentiment[vid]),\
               vid

    def __len__(self):
        return self.len 

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]