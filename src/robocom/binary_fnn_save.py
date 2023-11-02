import argparse
import pickle 
import yaml
import math
import datetime
import random
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle 
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

def load_data(test_vid, profile_vid):
    path = f'../../data/Hazumi_features/Hazumiall_features_binary.pkl'
    _, TS, _, _, Text, _, _, _ = pickle.load(open(path, 'rb'), encoding='utaf-8')   

    X_train, X_test, Y_train, Y_test = [], [], [], []

    for user in profile_vid:
        label = TS[user]
        data = Text[user]
        if user in test_vid:
            X_test.extend(data)
            Y_test.extend(label)
        else:
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
    fix_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=int, default=0) # 0: ベースライン, 1: 性別, 2: 年齢(2クラス), 3: 年齢(3クラス), 4: 年齢(6クラス) 5: 性別 & 年齢(2クラス), 6: 性別 & 年齢(3クラス), 7: 性別 & 年齢(6クラス)
    parser.add_argument('--balanced', action="store_true", default=False, help="class_weight is balanced")
    args = parser.parse_args()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

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

    results = []
    recalls = []

    for profile_n in range(profile_num):
        print(f"profile number : {profile_n}")
        profile_vid = []
        for id in vid:
            user_profile = profiling(id, args.profile)
            if user_profile == profile_n:
                profile_vid.append(id)
        print(f"profile group member : {len(profile_vid)}")

        '''
        1. データの準備
        '''
        test_vid = random.sample(profile_vid, int(len(profile_vid) * 0.25))
        print("test vid num = ", len(test_vid))
        x_train, y_train, x_valid, y_valid = load_data(test_vid, profile_vid)
        num_positive = np.sum(y_train)
        num_negative = y_train.size - num_positive
        pos_weight = torch.tensor(num_negative / num_positive)

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
            val_loss, preds = test_step(x_valid, y_valid)
            val_loss = val_loss.item() 

            if es(val_loss):
                break

        '''
        4. モデルの評価
        '''
        y_valid = y_valid.cpu().detach().numpy()
        preds = torch.sigmoid(preds)
        preds = preds.cpu().detach().numpy()

        fpr, tpr, thresholds = roc_curve(y_valid, preds)
        auc_score = round(auc(fpr, tpr), 3)

        results.append(auc_score)
        print("auc score : ", auc_score)
        preds = (preds >= 0.5).astype(int)
        cm = confusion_matrix(y_valid, preds)
        print("confusion matrix : \n", cm)
        print("low recall = ", round(cm[0][0] / sum(cm[0]), 3))
        print('-' * 30)
        recalls.append(round(cm[0][0] / sum(cm[0]), 3))

        '''
        5. モデルの保存 
        '''
        if args.balanced:
            torch.save(model.state_dict(), f'results/binary/fnn/model-balanced-{args.profile}-{profile_n}.pth')
        else:
            # torch.save(model.state_dict(), f'results/binary/fnn/model-{args.profile}-{profile_n}.pth')
            torch.save(model.state_dict(), f'results/binary/fnn/cpu-model-{args.profile}-{profile_n}.pth')
        
    print("================== Results ===================")
    print(f"results = ", results)
    print(f"recalls = ", recalls)
    print(round(sum(results) / len(results), 3))
    print(round(sum(recalls) / len(recalls), 3))
        