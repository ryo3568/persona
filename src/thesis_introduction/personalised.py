import argparse
import pickle 
import math
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn 
import torch.optim as optimizers 

import sys 
sys.path.append("../")

from model import FNNUniModel
from utils import torch_fix_seed, get_files
from utils.EarlyStopping import EarlyStopping


def load_data(test_user, modal, annot):
    path = f'../../data/Hazumi_features/Hazumi1911_features.pkl'
    SS, TS, _, _, Text, Audio, Visual, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    X_train = [] 
    Y_train = []
    X_test = []
    Y_test = []
    
    X_fine = []
    Y_fine = []

    for user in vid:
        if annot == "self":
            label = SS[user]
        else:
            label = TS[user]
        if 't' == modal:
            data = Text[user]
        elif 'a' == modal:
            audio = Audio[user]
            stds = StandardScaler()
            audio = stds.fit_transform(audio)
            data = audio
        if 'v' in modal:
            visual = Visual[user]
            stds = StandardScaler()
            visual = stds.fit_transform(visual)
            data = visual
        if user == test_user:
            x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
            X_train.extend(x_train)
            X_test.extend(x_test)
            Y_train.extend(y_train)
            Y_test.extend(y_test)

            X_fine.extend(x_train)
            Y_fine.extend(y_train)
        else:
            X_train.extend(data)
            Y_train.extend(label)
    

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_fine = np.array(X_fine)
    Y_fine = np.array(Y_fine)

    return X_train, Y_train, X_test, Y_test, X_fine, Y_fine

if __name__ == '__main__':
    '''
    0. 前準備
    '''
    torch_fix_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--modal', type=str, default="t")
    parser.add_argument('--label', type=str, default="self")
    args = parser.parse_args()
     
    if 't' == args.modal:
        input_dim = 768
    elif 'a' == args.modal:
        input_dim = 384
    else:
        input_dim = 66

    users = get_files("1911")

    results = []
    results_baseline = []

    for test_user in tqdm(users):
        '''
        1. データの準備
        '''
        x_train, y_train, x_test, y_test, x_fine, y_fine = load_data(test_user, args.modal, args.label)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)

        '''
        2. モデルの構築
        '''
        model = FNNUniModel(input_dim).to(device)
        
        '''
        3. モデルの学習
        '''
        criterion = nn.BCELoss() 
        optimizer = optimizers.Adam(model.parameters(), lr=0.0001)

        def train_step(x, y):
            model.train() 
            preds = model(x).reshape(-1)
            loss = criterion(preds, y)
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            return loss 
        
        def test_step(x, y):
            model.eval() 
            preds = model(x).reshape(-1)
            loss = criterion(preds, y)
            return loss, preds
        
        epochs = 100
        batch_size = 256
        n_batches = x_train.shape[0] // batch_size 

        es = EarlyStopping(patience=3)
            
        for epoch in range(epochs):
            train_loss = 0.
            x_, y_ = shuffle(x_train, y_train)
            x_ = torch.Tensor(x_).to(device)
            y_ = torch.Tensor(y_).to(device)

            for n_batch in range(n_batches):
                start = n_batch * batch_size 
                end = start + batch_size 
                loss = train_step(x_[start:end], y_[start:end])
                train_loss += loss.item() 

            x_valid = torch.Tensor(x_valid).to(device)
            y_valid = torch.Tensor(y_valid).to(device)
            val_loss, _ = test_step(x_valid, y_valid)
            val_loss = val_loss.item() 

            if es(val_loss):
                break

        '''
        4. モデルの評価
        '''
        
        x_test = torch.Tensor(x_test).to(device) 
        y_test = torch.Tensor(y_test).to(device) 
        _, preds = test_step(x_test, y_test)  
        preds = preds.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()

        fpr, tpr, thresholds = roc_curve(y_test, preds)

        # ROCの描画
        # plt.plot(fpr,tpr)
        # plt.xlabel('1-specificity (FPR)') 
        # plt.ylabel('sensitivity (TPR)')
        # plt.title('ROC Curve')
        # plt.show()

        # AUCの出力
        if not math.isnan(auc(fpr, tpr)):
            results_baseline.append(auc(fpr, tpr))

        '''
        5. モデルの保存 
        '''
        torch.save(model.state_dict(), f'results/pretrain/{test_user}-{args.label}-{args.modal}.pth')


    print("baseline : ", round(sum(results_baseline) / len(results_baseline), 3))

    