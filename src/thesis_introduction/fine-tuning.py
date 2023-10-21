import argparse
import pickle 
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn 
import torch.optim as optimizers 
import matplotlib.pyplot as plt

import sys 
sys.path.append("../")
from utils import get_files, torch_fix_seed
from utils.EarlyStopping import EarlyStopping
from model import GRUModel


def load_data(testuser, modal, gender):
    path = f'../../data/Hazumi_features/Hazumi1911_features.pkl'
    SS, _, _, _, Text, Audio, Visual, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    X_train = [] 
    Y_train = []
    
    for user in vid:
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
        if user == testuser:
            X_test = data 
            Y_test = label
        elif user[4] == gender:
            X_train.extend(data)
            Y_train.extend(label)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    '''
    0. 前準備
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--modal', type=str, default="tav")
    parser.add_argument('--tuning', type=int, default=-1)
    parser.add_argument('--gender', type=str, default="F")
    parser.add_argument('--pretrain', type=str, default="self")
    args = parser.parse_args()

    torch_fix_seed(123)
    users = get_files("1911")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = 0
    if 't' in args.modal:
        input_dim += 768
    if 'a' in args.modal:
        input_dim += 384
    if 'v' in args.modal:
        input_dim += 66

    for test_user in tqdm(users):
        if test_user[4] != args.gender:
            continue

        '''
        1. データの準備
        '''
        x_train, y_train, x_test, y_test = load_data(test_user, args.modal, args.gender)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)

        '''
        2. モデルの構築
        '''
        model = GRUModel(input_dim).to(device)
        model_name = "F" if args.gender == "M" else "M"
        model.load_state_dict(torch.load(f'results/pretrain/model-{args.pretrain}-{args.gender}.pth'))

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

        epochs = 50
        batch_size = 256
        n_batches = x_train.shape[0] // batch_size 

        es = EarlyStopping(patience=5)
        
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
                print(f"uid: {test_user}, epoch: {epoch} / {epochs}, early stopping")
                break

        '''
        4. モデルの評価
        '''
        x_test = torch.Tensor(x_test).to(device) 
        y_test = torch.Tensor(y_test).to(device)
        _, preds = test_step(x_test, y_test)  
        y_test = y_test.detach().cpu().numpy()
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

    