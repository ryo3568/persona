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
from utils import get_files, profiling, torch_fix_seed
from utils.EarlyStopping import EarlyStopping
from model import FNN


def load_data(testuser, modal, mode):
    path = f'../../data/Hazumi_features/Hazumi1911_features.pkl'
    SS, _, _, _, Text, Audio, Visual, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    X_train = [] 
    Y_train = []
    
    test_cluster = profiling(testuser, mode)

    for user in vid:
        user_cluster = profiling(user, mode)
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
        elif user_cluster == test_cluster:
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
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()

    torch_fix_seed(123)
    users = get_files("1911")
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
        x_train, y_train, x_test, y_test = load_data(test_user, args.modal, args.mode)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)

        '''
        2. モデルの構築
        '''
        model = FNN(input_dim, args.modal).to(device)
        if args.tuning >= 0:
            model.load_state_dict(torch.load(f'results/pretrain/model-{args.modal}-0.pth'))

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
        criterion = nn.CrossEntropyLoss() 
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
            model.eval() 
            preds = model(x) 
            loss = criterion(preds, y)
            preds = torch.argmax(preds, dim=1)
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
                val_losses.append(val_loss)

                if es(val_loss):
                    print(f"uid: {test_user}, epoch: {epoch} / {args.epochs}, early stopping")
                    break

        '''
        4. モデルの評価
        '''
        x_test = torch.Tensor(x_test).to(device) 
        y_test = torch.Tensor(y_test).long().to(device)
        loss, preds = test_step(x_test, y_test)  
        test_loss = loss.item() 
        y_test = y_test.cpu() 
        preds = preds.cpu()
        test_acc = round(accuracy_score(y_test, preds), 3)
        test_f1 = round(f1_score(y_test, preds, average="weighted"), 3)

        # print('test user: {}, test_acc: {:.3f}, test_f1: {:.3f}'.format(test_user, test_acc, test_f1))

        Acc.append(test_acc)
        F1.append(test_f1)
        Pred.extend(preds)
        Ans.extend(y_test)

        if args.plot:
            plot_loss(val_losses, test_acc, test_f1)

        '''
        5. モデルの保存 
        '''
        torch.save(model.state_dict(), f'results/finetuning/model-{args.modal}-{args.mode}-{device}.pth')
    
    print('========== Results ==========')
    print('acc: {:.3f}, f1: {:.3f}'.format(sum(Acc)/len(Acc), sum(F1)/len(F1)))

    print('========== Confusion Matrix =========')
    print(confusion_matrix(Ans, Pred))