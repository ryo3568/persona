import argparse
import pickle 
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, f1_score
import torch 
import torch.nn as nn 
import torch.optim as optimizers 

import sys 
sys.path.append("../")
from utils import get_files

class FNN(nn.Module):
    '''
    多層パーセプトロン
    '''
    def __init__(self, input_dim, modal):
        super().__init__() 

        unimodal_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU() ,
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        bimodal_stack = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU() ,
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU() ,
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU() ,
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        trimodal_stack = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU() ,
            nn.Dropout(0.3),
            nn.Linear(192, 64),
            nn.ReLU() ,
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        if len(modal) == 3:
            self.stack = trimodal_stack
        elif len(modal) == 2:
            self.stack = bimodal_stack
        else:
            self.stack = unimodal_stack

    def forward(self, x):
        y = self.stack(x) 
        return y

def clustering(id, TP):
    return 1

def load_data(modal, version):
    # path = f'../../data/Hazumi_features/Hazumi{version}_features.pkl'
    path = f'../../data/Hazumi_features/Hazumiall_features.pkl'
    SS, TS, _, TP, Text, Audio, Visual, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    X_train = [] 
    Y_train = []
    X_test = []
    Y_test = []
    
    # test_cluster = clustering(testuser, TP)

    X = [] 
    Y = []

    for user in vid:
        # user_cluster = clustering(user, TP)
        label = pd.DataFrame(TS[user])
        data = [] 
        if 't' in modal:
            text = pd.DataFrame(Text[user])
            data.append(text)
        if 'a' in modal:
            audio = Audio[user]
            stds = StandardScaler()
            audio = stds.fit_transform(audio)
            audio = pd.DataFrame(audio)
            data.append(audio)
        if 'v' in modal:
            visual = Visual[user]
            stds = StandardScaler()
            visual = stds.fit_transform(visual)
            visual = pd.DataFrame(visual)
            data.append(visual)
        data = pd.concat(data, axis=1)
        X.append(data)
        Y.append(label)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    X_train = pd.concat(X_train).values
    Y_train = pd.concat(Y_train).values
    X_test = pd.concat(X_test).values
    Y_test = pd.concat(Y_test).values

    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    '''
    0. 前準備
    '''
    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="1911")
    parser.add_argument('--modal', type=str, default="tav")
    args = parser.parse_args()

    users = get_files(args.version)

    Acc = []
    F1 = []

    input_dim = 0
    if 't' in args.modal:
        input_dim += 768
    if 'a' in args.modal:
        input_dim += 384
    if 'v' in args.modal:
        input_dim += 66
    
    '''
    1. データの準備
    '''
    x_train, y_train, x_test, y_test = load_data(args.modal, args.version)

    '''
    2. モデルの構築
    '''
    model = FNN(input_dim, args.modal).to(device)
    
    '''
    3. モデルの学習
    '''
    criterion = nn.BCELoss() 
    optimizer = optimizers.Adam(model.parameters(), lr=0.001)

    def train_step(x, y):
        model.train() 
        preds = model(x) 
        loss = criterion(preds, y)
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        return loss 
    
    epochs = 30
    batch_size = 32
    n_batches = x_train.shape[0] // batch_size 
        
    for epoch in tqdm(range(epochs)):
        train_loss = 0.
        x_, y_ = shuffle(x_train, y_train)
        x_ = torch.Tensor(x_).to(device)
        y_ = torch.Tensor(y_).to(device).reshape(-1, 1)

        for n_batch in range(n_batches):
            start = n_batch * batch_size 
            end = start + batch_size 
            loss = train_step(x_[start:end], y_[start:end])
            train_loss += loss.item() 

        # print('epoch: {}, loss: {:.3}'.format(epoch+1, train_loss))

    '''
    4. モデルの評価
    '''
    def test_step(x, y):
        x = torch.Tensor(x).to(device) 
        y = torch.Tensor(y).to(device).reshape(-1, 1)
        model.eval() 
        preds = model(x) 
        loss = criterion(preds, y)
        return loss, preds 
    
    loss, preds = test_step(x_test, y_test)  
    test_loss = loss.item() 
    preds = (preds.data.cpu().numpy() > 0.5).astype(int).reshape(-1)
    test_acc = accuracy_score(y_test, preds) 
    test_f1 = f1_score(y_test, preds)

    print('test_acc: {:.3f}, test_f1: {:.3f}'.format(test_acc, test_f1))

    '''
    5. モデルの保存 
    '''
    torch.save(model.state_dict(), f'results/model/{args.modal}/model.pth')
    