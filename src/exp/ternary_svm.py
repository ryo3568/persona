import argparse
import pickle 
import yaml
import math
import datetime
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle 
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import torch 
import torch.nn as nn 
import torch.optim as optimizers 

import sys 
sys.path.append("../")
from utils import get_files, profiling, torch_fix_seed
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
    parser.add_argument('--balanced', action="store_true", default=False, help="class_weight is balanced")
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

        '''
        2. モデルの構築
        '''
        if args.balanced:
            model = svm.SVC(C=100, gamma=0.0001, kernel="sigmoid", class_weight="balanced")
        else:
            model = svm.SVC(C=100, gamma=0.0001, kernel="sigmoid")

        '''
        3. モデルの学習
        '''
        model.fit(x_train, y_train)

        '''
        4. モデルの評価
        '''
        preds = model.predict(x_test)

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
    if args.balanced:
        output_filename = f'results/ternary/balanced/{args.profile}-' + now.strftime('%Y%m%d_%H%M%S') + '.yaml'
    else:
        output_filename = f'results/ternary/svm/{args.profile}-' + now.strftime('%Y%m%d_%H%M%S') + '.yaml'

    with open(output_filename, 'w') as file:
        yaml.dump(results, file)