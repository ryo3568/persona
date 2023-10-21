import argparse
import pickle 
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn 
import torch.optim as optimizers 
import matplotlib.pyplot as plt

import sys 
sys.path.append("../")
from utils import get_files
from utils.EarlyStopping import EarlyStopping
from model import FNN


def clustering(id, mode):
    age = int(id[5])
    gender = id[4]

    if mode == 0:
        res = 0
    elif mode == 1:
        # 性別
        if gender == 'F':
            res = 0
        else: res = 1
    elif mode == 2:
        # 年齢 40 <=, 40 >
        if age <= 4:
            res = 0
        else:
            res = 1
    elif mode == 3:
        # 年齢 30 <=, 50 <=, 50 > 
        if age <= 3:
            res = 0 
        elif age <= 5:
            res = 1
        else:
            res = 2
    elif mode == 4: 
        # 年齢 20, 30, 40, 50, 60, 70
        res = age - 2
    return res

def load_data(testuser, modal, version, mode):
    path = f'../../data/Hazumi_features/Hazumi{version}_features.pkl'
    SS, TS, _, TP, Text, Audio, Visual, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    X_train = [] 
    Y_train = []
    X_test = []
    Y_test = []
    
    test_cluster = clustering(testuser, mode)

    for user in vid:
        user_cluster = clustering(user, mode)
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

    return X_train, Y_train, X_test, Y_test, test_cluster

if __name__ == '__main__':
    '''
    0. 前準備
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="1911")
    parser.add_argument('--modal', type=str, default="tav")
    parser.add_argument('--tuning', type=int, default=-1)
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()

    users = get_files(args.version)
    seed_num = args.seed
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Acc = []
    F1 = []
    Pred = []
    Ans = []

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
        x_train, y_train, x_test, y_test, test_gclass = load_data(test_user, args.modal, args.version, args.mode)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)

        '''
        2. モデルの構築
        '''
        model = FNN(input_dim, args.modal).to(device)
        if args.tuning >= 0:
            mode_path = ["all", "gender", "age2", "age3", "age6"]
            model.load_state_dict(torch.load(f'results/model/{mode_path[args.mode]}/{args.modal}/model-{test_gclass}.pth'))
            # model.load_state_dict(torch.load(f'results/model/all/{args.modal}/model-cpu.pth'))

            for param in model.parameters():
                param.requires_grad = False

            if args.modal == 't': 
                linear_num = [12, 9, 6, 3, 0]
            else:
                linear_num = [15, 12, 9, 6, 3, 0]

            for i in range(args.tuning):
                for param in model.stack[linear_num[i]].parameters():
                    param.requires_grad = True

        '''
        3. モデルの学習
        '''
        criterion = nn.BCELoss() 
        optimizer = optimizers.Adam(model.parameters(), lr=args.lr)

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

        def plot_loss(loss, acc, f1):
            sizes = [i for i in range(len(loss))]
            plt.figure()
            plt.title(f"user id : {test_user}, Acc: {acc}, F1: {f1}")
            plt.xlabel("epoch")
            plt.ylabel("BCE Loss")
            plt.plot(sizes, loss, 'o-', color="r", label="Train")
            plt.legend(loc="best")
            plt.show()
        
        epochs = args.epochs
        batch_size = 32
        n_batches = x_train.shape[0] // batch_size 
        val_losses = []

        es = EarlyStopping(patience=5)
        
        if args.tuning != 0:
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

                val_loss, _ = test_step(x_valid, y_valid)
                val_loss = val_loss.item() 
                val_losses.append(val_loss)

                if es(val_loss):
                    print(f"uid: {test_user}, epoch: {epoch} / {args.epochs}, early stopping")
                    break

        '''
        4. モデルの評価
        '''
        loss, preds = test_step(x_test, y_test)  
        test_loss = loss.item() 
        preds = (preds.data.cpu().numpy() >= 0.5).astype(int).reshape(-1)
        test_acc = round(accuracy_score(y_test, preds), 3)
        test_f1 = round(f1_score(y_test, preds), 3)

        # print('test user: {}, test_acc: {:.3f}, test_f1: {:.3f}'.format(test_user, test_acc, test_f1))

        Acc.append(test_acc)
        F1.append(test_f1)
        Pred.extend(preds)
        Ans.extend(y_test)

        cluster = clustering(test_user, args.mode)

        if args.plot:
            plot_loss(val_losses, test_acc, test_f1)

        '''
        5. モデルの保存 
        '''
        torch.save(model.state_dict(), f'results/model/all/model-cpu.pth')
    
    print('========== Results ==========')
    print('acc: {:.3f}, f1: {:.3f}'.format(sum(Acc)/len(Acc), sum(F1)/len(F1)))

    print('========== Confusion Matrix =========')
    Ans1 = [] 
    Pred1 = []
    for ans, pred in zip(Ans, Pred):
        ans = 1 if ans == 0 else 0 
        pred = 1 if pred == 0 else 0
        Ans1.append(ans)
        Pred1.append(pred)

    print(confusion_matrix(Ans1, Pred1))
    print(round(f1_score(Ans1, Pred1), 3))