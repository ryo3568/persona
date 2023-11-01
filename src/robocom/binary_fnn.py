import argparse
import pickle 
import yaml
import math
import datetime
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle 
from sklearn.model_selection import KFold 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import torch 
import torch.nn as nn 
import torch.optim as optimizers 

import sys 
sys.path.append("../")
from utils import profiling, fix_seed
from model import FNNUniModel
from utils.EarlyStopping import EarlyStopping

def load_data(train, profile_vid):
    path = f'../../data/Hazumi_features/Hazumiall_features_binary.pkl'
    _, TS, _, _, Text, _, _, _ = pickle.load(open(path, 'rb'), encoding='utf-8')

    X_train = [] 
    Y_train = []

    for i, user in enumerate(profile_vid):
        label = TS[user]
        data = Text[user]
        if i in train:
            X_train.extend(data)
            Y_train.extend(label)
            X_test = data
            Y_test = label
        else:
            X_test = data 
            Y_test = label
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test
    

if __name__ == '__main__':
    '''
    0. 前準備
    '''
    fix_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=int, default=0) # 0: ベースライン, 1: 性別, 2: 年齢(2クラス), 3: 年齢(3クラス), 4: 年齢(6クラス) 5: 性別 & 年齢(2クラス), 6: 性別 & 年齢(3クラス), 7: 性別 & 年齢(6クラス)
    parser.add_argument('--balanced', action="store_true", default=False, help="class_weight is balanced")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = f'../../data/Hazumi_features/Hazumiall_features_binary.pkl'
    _, _, _, _, _, _, _, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    if args.profile == 0:
        profile_num = 1 
    elif args.profile == 1:
        profile_num = 2 
    elif args.profile == 2:
        profile_num = 2
    elif args.profile == 3:
        profile_num = 3
    elif args.profile == 4:
        profile_num = 6
    elif args.profile == 5:
        profile_num = 4 
    elif args.profile == 6:
        profile_num = 6
    else:
        profile_num = 12

    Results = []

    kf = KFold(n_splits=3, shuffle=True, random_state=1)

    for profile_n in range(profile_num):
        print(f"profile number : {profile_n}")
        profile_vid = []
        for id in vid:
            user_profile = profiling(id, args.profile)
            if user_profile == profile_n:
                profile_vid.append(id)
        print(f"profile group member : {len(profile_vid)}")
        results = []

        for train, test in kf.split(profile_vid):
            '''
            1. データの準備
            '''
            x_train, y_train, x_test, y_test = load_data(train, profile_vid)
            num_positive = np.sum(y_train)
            num_negative = y_train.size - num_positive
            pos_weight = torch.tensor(num_negative / num_positive)
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)

            '''
            2. モデルの構築
            '''
            model = FNNUniModel(768).to(device)

            '''
            3. モデルの学習
            '''
            if args.balanced:
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
            else:
                criterion = nn.BCEWithLogitsLoss() 
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
            batch_size = 128
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
                    break

            '''
            4. モデルの評価
            '''
            x_test = torch.Tensor(x_test).to(device) 
            y_test = torch.Tensor(y_test).to(device)
            loss, preds = test_step(x_test, y_test)  
            y_test = y_test.cpu().detach().numpy()
            preds = preds.cpu().detach().numpy()
            
            fpr, tpr, thresholds = roc_curve(y_test, preds)
            auc_score = auc(fpr, tpr)
            # preds = (preds >= 0.5).astype(int)
            # acc = accuracy_score(y_test, preds) 
            # f1 = f1_score(y_test, preds, average="weighted")
            # cm = confusion_matrix(y_test, preds)


            results.append(auc_score)
            print("auc score : ", auc_score)
            # preds = (preds > 0.000).astype(int)
            preds = (preds >= 0.5).astype(int)
            print("confusion matrix : \n", confusion_matrix(y_test, preds))
        
        Results.append(round(sum(results) / len(results), 3))

    print('========== Results ==========')
    print(Results)
    Results_clean = []
    for result in Results:
        if not math.isnan(result):
            Results_clean.append(result)
    print(round(sum(Results_clean) / len(Results_clean), 3))