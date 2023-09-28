import argparse
import pickle 
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn 
import torch.optim as optimizers 
import matplotlib.pyplot as plt

import sys 
sys.path.append("../")
from utils import get_files
from model import FNN


def clustering(id, TP):
    age = int(id[5])
    gender = id[4]

    # 性別
    # if gender == 'F':
    #     res = 0
    # else:
    #     res = 1

    # 年齢 40 <=, 40 >
    # if age > 4:
    #     res = 0
    # else:
    #     res = 1

    # 年齢 30 <=, 50 <=, 50 > 
    # if age <= 3:
    #     res = 0 
    # elif age <= 5:
    #     res = 1
    # else:
    #     res = 2
    
    # 年齢 20, 30, 40, 50, 60, 70
    res = age

    return res

def load_data(testuser, modal, version):
    path = f'../../data/Hazumi_features/Hazumi{version}_features.pkl'
    SS, TS, _, TP, Text, Audio, Visual, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    X_train = [] 
    Y_train = []
    X_test = []
    Y_test = []
    
    test_cluster = clustering(testuser, TP)

    for user in vid:
        user_cluster = clustering(user, TP)
        label = pd.DataFrame(SS[user])
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
        if user == testuser:
            X_test.append(data)
            Y_test.append(label)
        elif user_cluster == test_cluster:
            X_train.append(data)
            Y_train.append(label)

    X_train = pd.concat(X_train).values
    Y_train = pd.concat(Y_train).values
    X_test = pd.concat(X_test).values
    Y_test = pd.concat(Y_test).values

    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    '''
    0. 前準備
    '''
    seed_num = 122
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
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
    for test_user in tqdm(users):
        '''
        1. データの準備
        '''
        x_train, y_train, x_test, y_test = load_data(test_user, args.modal, args.version)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)

        '''
        2. モデルの構築
        '''
        model = FNN(input_dim, args.modal).to(device)
        model.load_state_dict(torch.load(f'results/model/{args.modal}/model.pth'))


        for param in model.parameters():
            param.requires_grad = False

        # for param in model.stack[0].parameters():
        #     param.requires_grad = True 

        # for param in model.stack[3].parameters():
        #     param.requires_grad = True 

        # for param in model.stack[6].parameters():
        #     param.requires_grad = True 

        for param in model.stack[9].parameters():
            param.requires_grad = True 

        for param in model.stack[12].parameters():
            param.requires_grad = True 

        for param in model.stack[15].parameters():
            param.requires_grad = True 


        '''
        3. モデルの学習
        '''
        criterion = nn.BCELoss() 
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
            x = torch.Tensor(x).to(device) 
            y = torch.Tensor(y).to(device).reshape(-1, 1)
            model.eval() 
            preds = model(x) 
            loss = criterion(preds, y)
            return loss, preds 
        
        epochs = 30
        batch_size = 32
        n_batches = x_train.shape[0] // batch_size 
        Acc = []
        F1 = []
            
        for epoch in range(epochs):
            train_loss = 0.
            x_, y_ = shuffle(x_train, y_train)
            x_ = torch.Tensor(x_).to(device)
            y_ = torch.Tensor(y_).to(device).reshape(-1, 1)

            for n_batch in range(n_batches):
                start = n_batch * batch_size 
                end = start + batch_size 
                loss = train_step(x_[start:end], y_[start:end])
                train_loss += loss.item() 

            loss, preds = test_step(x_valid, y_valid)
            valid_loss = loss.item() 
            preds = (preds.data.cpu().numpy() > 0.5).astype(int).reshape(-1)
            valid_acc = accuracy_score(y_valid, preds) 
            Acc.append(valid_acc)
            valid_f1 = f1_score(y_valid, preds)
            F1.append(valid_f1)
            # print('epoch: {}, loss: {:.3}'.format(epoch+1, train_loss))
        

        def plot_loss(loss):
            sizes = [i for i in range(len(loss))]
            plt.figure()
            plt.title(f"user id : {test_user}")
            plt.xlabel("epoch")
            plt.ylabel("BCE Loss")
            plt.plot(sizes, loss, 'o-', color="r", label="Train")
            plt.legend(loc="best")
            plt.show()
        
        # plot_loss(Acc)

        '''
        4. モデルの評価
        '''
        loss, preds = test_step(x_test, y_test)  
        test_loss = loss.item() 
        preds = (preds.data.cpu().numpy() > 0.5).astype(int).reshape(-1)
        test_acc = accuracy_score(y_test, preds) 
        test_f1 = f1_score(y_test, preds)

        # print('test user: {}, test_acc: {:.3f}, test_f1: {:.3f}'.format(test_user, test_acc, test_f1))

        Acc.append(test_acc)
        F1.append(test_f1)

        '''
        5. モデルの保存 
        '''
        # torch.save(model.state_dict(), f'../data/model/model_weight{args.version}.pth')
    
    print('========== Results ==========')
    print('acc: {:.3f}, f1: {:.3f}'.format(sum(Acc)/len(Acc), sum(F1)/len(F1)))