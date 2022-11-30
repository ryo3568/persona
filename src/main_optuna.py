import os 
import sys
sys.path.append('../')
import glob 
import pickle
from tqdm import tqdm 

import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler 
from torch.nn.utils.rnn import pad_sequence 

import optuna 

from model import LSTMModel
from utils.Standardizing import Standardizing

import warnings 
warnings.simplefilter('ignore')

class HazumiDataset(Dataset):

    def __init__(self, test_file, train=True, scaler=None, args=None):
    
        path = '../data/Hazumi_features/Hazumi1911_features.pkl'


        self.SS_ternary, self.TS_ternary, self.sentiment, self.third_sentiment, self.persona, self.third_persona,\
        self.text, self.audio, self.visual, self.vid = pickle.load(open(path, 'rb'), encoding='utf-8')

        self.keys = [] 

        self.args = args

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

        
    def __getitem__(self, index):
        vid = self.keys[index] 

        sentiment = self.third_sentiment
        s_ternary = self.TS_ternary

        persona = self.third_persona

        
        return torch.FloatTensor(self.text[vid]),\
            torch.FloatTensor(self.scaler_visual.transform(self.visual[vid])),\
            torch.FloatTensor(self.scaler_audio.transform(self.audio[vid])),\
            torch.FloatTensor(persona[vid]),\
            torch.FloatTensor(sentiment[vid]),\
            torch.LongTensor(s_ternary[vid]),\
            vid

    def __len__(self):
        return self.len 

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

def trial_optimizer(trial, model):
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-10, 1e-3)
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
    optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)

    return optimizer

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset) 
    idx = list(range(size)) 
    split = int(valid*size) 
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_Hazumi_loaders(test_file, batch_size=32, valid=0.1, args=None, num_workers=2, pin_memory=False):
    trainset = HazumiDataset(test_file, args=args)
    testset = HazumiDataset(test_file, train=False, scaler=trainset.scaler, args=args) 

    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset, 
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader 

def train_or_eval_model(model, loss_function, dataloader, optimizer=None, train=False, device=None):
    losses = []
    assert not train or optimizer!=None 
    if train:
        model.train() 
    else:
        model.eval() 

    for data in dataloader:
        if train:
            optimizer.zero_grad() 
        
        text, visual, audio, persona, sentiment, s_ternary =\
        [d.cuda() for d in data[:-1]] if device == 'cuda' else data[:-1]

        # data = audio
        # data = torch.cat((visual, audio), dim=-1)
        data = torch.cat((text, visual, audio), dim=-1)
            
        pred = model(data)

        loss = loss_function(pred, persona)      

        # 学習ログ
        losses.append(loss.item())

        if train:
            loss.backward()
            optimizer.step() 

    avg_loss = round(np.sum(losses)/len(losses), 4)

    return avg_loss 



def objective(trial):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    testfiles = []
    for f in glob.glob('../data/Hazumi1911/dumpfiles/*.csv'):
        testfiles.append(os.path.splitext(os.path.basename(f))[0])

    testfiles = sorted(testfiles)

    loss = [] 

    D_i = 3063

    # 中間層のユニット数の試行
    D_h = int(trial.suggest_discrete_uniform("D_h", 300, 500, 100))
    D_o = int(trial.suggest_discrete_uniform("D_o", 32, 256, 32))

    # drop-out rateの試行　
    in_droprate = trial.suggest_discrete_uniform("in_droprate", 0.0, 0.2, 0.05)
    for testfile in tqdm(testfiles, position=0, leave=True):

        model = LSTMModel(D_i, D_h, D_o, n_classes=5, dropout=in_droprate).to(device)
        loss_function = nn.MSELoss() 
        optimizer = trial_optimizer(trial, model)
        train_loader, valid_loader, test_loader = get_Hazumi_loaders(testfile, batch_size=BATCH_SIZE, valid=0.1) 

        best_loss = None 

        for epoch in range(EPOCH):
            train_or_eval_model(model, loss_function, train_loader, optimizer, True, device=device)
            train_or_eval_model(model, loss_function, valid_loader, device=device)
            tst_loss = train_or_eval_model(model, loss_function, test_loader, device=device)

            if best_loss == None or best_loss > tst_loss:
                best_loss = tst_loss 
        
        loss.append(best_loss)

    return sum(loss) / len(loss)

if __name__ == '__main__':
    # 実験設定
    TRIAL_SIZE = 1
    BATCH_SIZE = 1
    EPOCH = 60 
    N_CLASS = 5 

    # ハイパラチューニング
    study = optuna.create_study() 
    study.optimize(objective, n_trials=TRIAL_SIZE)

    print(study.best_params)