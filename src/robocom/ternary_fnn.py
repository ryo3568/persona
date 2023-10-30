import argparse
import pickle 
import yaml
import math
import datetime
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import torch 
import torch.nn as nn 
import torch.optim as optimizers 

import sys 
sys.path.append("../")
from utils import get_files, profiling, torch_fix_seed
from model import FNNTernaryModel
from utils.EarlyStopping import EarlyStopping

def load_data(testuser, modal, profile):
    path = f'../../data/Hazumi_features/Hazumi1911_features_ternary.pkl'
    _, TS, _, _, Text, Audio, Visual, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    X_train = [] 
    Y_train = []
    
    test_profile = profiling(testuser, profile)

    for user in vid:
        user_profile = profiling(user, profile)
        label = TS[user]
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
        elif user_profile == test_profile:
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
    torch_fix_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--modal', type=str, default="t")
    parser.add_argument('--profile', type=int, default=0) # 0: ベースライン, 1: 性別, 2: 年齢(2クラス), 3: 年齢(3クラス), 4: 年齢(6クラス) 5: 性別 & 年齢(2クラス), 6: 性別 & 年齢(3クラス), 7: 性別 & 年齢(6クラス)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    users = get_files("1911")

    input_dim = 0
    if 't' in args.modal:
        input_dim += 768
    if 'a' in args.modal:
        input_dim += 384
    if 'v' in args.modal:
        input_dim += 66

    results = {}

    for test_user in tqdm(users):
        '''
        1. データの準備
        '''
        x_train, y_train, x_test, y_test = load_data(test_user, args.modal, args.profile)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)

        '''
        2. モデルの構築
        '''
        model = FNNTernaryModel(input_dim).to(device)

        '''
        3. モデルの学習
        '''
        criterion = nn.CrossEntropyLoss() 
        optimizer = optimizers.Adam(model.parameters(), lr=0.0001)

        def train_step(x, y):
            model.train() 
            preds = model(x)
            loss = criterion(preds, y)
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            return loss 

        def test_step(x, y):
            model.eval() 
            preds = model(x)
            loss = criterion(preds, y)
            preds = torch.argmax(preds, dim=1)
            return loss, preds 
        
        epochs = 100
        batch_size = 128
        n_batches = x_train.shape[0] // batch_size 

        es = EarlyStopping(patience=5)
            
        for epoch in range(epochs):
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
            y_valid = torch.Tensor(y_valid).long().to(device)
            val_loss, _ = test_step(x_valid, y_valid)
            val_loss = val_loss.item() 

            if es(val_loss):
                break

        '''
        4. モデルの評価
        '''
        x_test = torch.Tensor(x_test).to(device) 
        y_test = torch.Tensor(y_test).long().to(device)
        loss, preds = test_step(x_test, y_test)  
        y_test = y_test.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
        
        acc = accuracy_score(y_test, preds) 
        f1 = f1_score(y_test, preds, average="weighted")
        cm = confusion_matrix(y_test, preds)

        print('test user: {}, acc: {:.3f}, f1: {:.3f}'.format(test_user, acc, f1))

        results[test_user] = {
            'acc': float(round(acc, 3)),
            'f1': float(round(f1, 3)),
            'confusion matrix': cm.tolist()
        }

    print('========== Results ==========')
    ACC, F1 = [], []
    for uid in results:
        ACC.append(results[uid]["acc"])
        F1.append(results[uid]["f1"])
        
    results["all"] = {
        'acc': round(sum(ACC) / len(ACC), 3),
        'f1': round(sum(F1) / len(F1), 3),
    }
    print(results)

    now = datetime.datetime.now()
    output_filename = f'results/ternary/fnn/{args.profile}-' + now.strftime('%Y%m%d_%H%M%S') + '.yaml'
    with open(output_filename, 'w') as file:
        yaml.dump(results, file)
    # print('acc: {:.3f}, f1: {:.3f}'.format(sum(Acc)/len(Acc), sum(F1)/len(F1)))