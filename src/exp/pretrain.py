import argparse
import pickle 
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch 
import torch.nn as nn 
import torch.optim as optimizers 

import sys 
sys.path.append("../")

from model import FNNUniModel
from utils import torch_fix_seed
from utils.EarlyStopping import EarlyStopping


def load_data(modal):
    path = f'../../data/Hazumi_features/Hazumiall_features.pkl'
    _, TS, _, _, Text, Audio, Visual, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    X = [] 
    Y = []

    for user in vid:
        label = TS[user]
        if 't' == modal:
            data = Text[user]
        if 'a' == modal:
            audio = Audio[user]
            stds = StandardScaler()
            audio = stds.fit_transform(audio)
            data = audio
        if 'v' == modal:
            visual = Visual[user]
            stds = StandardScaler()
            visual = stds.fit_transform(visual)
            data = visual
        X.extend(data)
        Y.extend(label)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    '''
    0. 前準備
    '''
    torch_fix_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--modal', type=str, default="t")
    args = parser.parse_args()
     
    results = []

    if 't' in args.modal:
        input_dim = 768
    if 'a' in args.modal:
        input_dim = 384
    if 'v' in args.modal:
        input_dim = 66
    
    '''
    1. データの準備
    '''
    x_train, y_train, x_test, y_test = load_data(args.modal)
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
    
    epochs = 100
    batch_size = 128
    n_batches = x_train.shape[0] // batch_size 

    es = EarlyStopping(patience=5)
        
    for epoch in tqdm(range(epochs)):
        train_loss = 0.
        x_, y_ = shuffle(x_train, y_train)
        x_ = torch.Tensor(x_).to(device)
        y_ = torch.Tensor(y_).long().to(device)

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
    def test_step(x, y):
        model.eval() 
        preds = model(x) 
        loss = criterion(preds, y)
        preds = torch.argmax(preds, dim=1)
        return loss, preds 
    
    x_test = torch.Tensor(x_test).to(device) 
    y_test = torch.Tensor(y_test).long().to(device)
    loss, preds = test_step(x_test, y_test)  
    test_loss = loss.item() 
    y_test = y_test.cpu()
    preds = preds.cpu()

    test_acc = accuracy_score(y_test, preds) 
    test_f1 = f1_score(y_test, preds, average="weighted")

    print('test_acc: {:.3f}, test_f1: {:.3f}'.format(test_acc, test_f1))
    print('confusion matrix\n', confusion_matrix(y_test, preds))

    '''
    5. モデルの保存 
    '''
    torch.save(model.state_dict(), f'results/pretrain/model-{args.modal}-{args.mode}.pth')

    