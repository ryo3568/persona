import argparse
import pickle 
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

from model import GRUModel
from utils import torch_fix_seed
from utils.EarlyStopping import EarlyStopping


def load_data(modal, gender):
    path = f'../../data/Hazumi_features/Hazumi1911_features.pkl'
    SS, _, _, _, Text, Audio, Visual, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    X = [] 
    Y = []

    for user in vid:
        if user[4] != gender:
            continue
        label = SS[user]
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
        data = pd.concat(data, axis=1).values
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
    parser.add_argument('--modal', type=str, default="tav")
    parser.add_argument('--gender', type=str, default='F')
    args = parser.parse_args()
     
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
    x_train, y_train, x_test, y_test = load_data(args.modal, args.gender)

    '''
    2. モデルの構築
    '''
    model = GRUModel(input_dim).to(device)
    
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
    
    epochs = 50
    batch_size = 256
    n_batches = x_train.shape[0] // batch_size 
        
    for epoch in tqdm(range(epochs)):
        train_loss = 0.
        x_, y_ = shuffle(x_train, y_train)
        x_ = torch.Tensor(x_).to(device)
        y_ = torch.Tensor(y_).to(device)

        for n_batch in range(n_batches):
            start = n_batch * batch_size 
            end = start + batch_size 
            loss = train_step(x_[start:end], y_[start:end])
            train_loss += loss.item() 

    '''
    4. モデルの評価
    '''
    def test_step(x):
        model.eval() 
        preds = model(x).reshape(-1)
        return preds
    
    x_test = torch.Tensor(x_test).to(device) 
    preds = test_step(x_test)  
    preds = preds.detach().cpu().numpy()

    fpr, tpr, thresholds = roc_curve(y_test, preds)

    # ROCの描画
    plt.plot(fpr,tpr)
    plt.xlabel('1-specificity (FPR)') 
    plt.ylabel('sensitivity (TPR)')
    plt.title('ROC Curve')
    plt.show()

    # AUCの出力
    print("acu: ", auc(fpr, tpr))

    '''
    5. モデルの保存 
    '''
    torch.save(model.state_dict(), f'results/pretrain/model-self-{args.gender}.pth')

    