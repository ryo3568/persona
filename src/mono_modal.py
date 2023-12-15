import os
import pickle
import argparse
import yaml
import datetime
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch 
import torch.nn as nn 
import torch.optim as optim 

import warnings
warnings.filterwarnings('ignore')

from dataset import HazumiDataset_torch, HazumiTestDataset_torch
from model import UnimodalFNN
from utils.EarlyStopping import EarlyStopping
from utils import fix_seed, profiling


def train_fn(model, train_loader, criterion, optimizer):

    # 1 epoch training 
    train_loss = 0.0 
    num_train = 0 

    # model 学習モードに設定
    model.train() 

    for x, y in train_loader:
        # batch数の累積
        num_train += len(y)

        x, y = x.to(device), y.view(-1).to(device)
        # 勾配をリセット
        optimizer.zero_grad() 
        # 推論
        output = model(x)
        # lossの計算
        if args.binary:
            loss = criterion(output.view(-1), y.float())
        else:
            loss = criterion(output, y)
        # 誤差逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step() 
        # lossを累積
        train_loss += loss.item() 
    
    train_loss = train_loss / num_train 

    return train_loss

def valid_fn(model, test_loader, criterion):

    # 評価用のコード
    valid_loss = 0.0 
    num_valid = 0 
    outputs = []
    labels = []

    # model　評価モードに設定
    model.eval() 

    # 評価の際に勾配を計算しないようにする
    with torch.no_grad():
        for x, y in test_loader:
            num_valid += len(y)
            x, y = x.to(device), y.view(-1).to(device) 
            output = model(x)
            # loss = criterion(output, y)
            if args.binary:
                loss = criterion(output.view(-1), y.float())
            else:
                loss = criterion(output, y)
            valid_loss += loss.item()
            # outputの蓄積
            if args.binary:
                _output = (output >= 0.5).int()
            else:
                _output = torch.argmax(output, dim=1)
            outputs.extend(_output.tolist())
            # labelの蓄積
            labels.extend(y.tolist())

        valid_loss = valid_loss / num_valid

        if args.regression:
            res = valid_loss
        else:
            res = accuracy_score(labels, outputs)

    return valid_loss, res


def run(model, train_loader, valid_loader, criterion, optimizer):

    train_loss_list = [] 
    valid_loss_list = [] 

    es = EarlyStopping(patience=10)

    # epoch = 1
    # while True:
    for epoch in range(num_epochs):
        _train_loss = train_fn(model, train_loader, criterion, optimizer)
        _valid_loss, _valid_acc = valid_fn(model, valid_loader, criterion)

        print(f'Epoch [{epoch+1}], train_loss: {_train_loss:.5f}, val_loss: {_valid_loss:.5f}, val_acc: {_valid_acc:.5f}')

        train_loss_list.append(_train_loss)
        valid_loss_list.append(_valid_loss)

        if es(_valid_loss):
            break
        
        epoch += 1
    
    return train_loss_list, valid_loss_list

def show_plot(train_list, test_list, xlabel="epoch", ylabel="loss", title="training and validation loss"):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(range(len(train_list)), train_list, c='b', label="train loss")
    ax.plot(range(len(test_list)), test_list, c='r', label="test loss")


    ax.set_xlabel(xlabel, fontsize='20')
    ax.set_ylabel(ylabel, fontsize='20')
    ax.set_title(title, fontsize='20')
    ax.grid() 
    ax.legend(fontsize="20")

    plt.show()


def evaluation(model, sscaler, id, criterion):
    test_dataset = HazumiTestDataset_torch(version=hazumi_version, id=id, sscaler=sscaler, modal=args.modal, ss=args.ss, binary=args.binary, regression=args.regression)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    # 評価用のコード
    num_test = 0 
    outputs = []
    labels = []

    loss = 0

    # model　評価モードに設定
    model.eval() 

    # 評価の際に勾配を計算しないようにする
    with torch.no_grad():
        for x, y in test_loader:
            num_test += len(y)
            x, y = x.to(device), y.view(-1).to(device) 
            output = model(x)
            # outputの蓄積
            if args.binary:
                _output = (output >= 0.5).int()
                outputs.extend(_output.tolist())
            elif args.regression:
                loss += criterion(output, y)
            else:
                _output = torch.argmax(output, dim=1)
                outputs.extend(_output.tolist())
            # _output = torch.argmax(output, dim=1)
            # outputs.extend(_output.tolist())
            # labelの蓄積
            labels.extend(y.tolist())
        if args.regression:
            loss /= num_test
            results[id] = float(loss)
        else:
            acc = accuracy_score(labels, outputs)
            results[id] = float(round(acc, 3))

def save_results():
    # 実験結果のファイル出力
    config = {}
    # config["fs"] = args.fs 
    config["annot"] = "SS" if args.ss else "TS"
    config["label"] = "binary" if args.binary else "ternary"
    config["modal"] = args.modal
    config["pmode"] = args.pmode
    config["task"] = "regresion" if args.regression else "classification"
    
    yml = {}
    yml["config"] = config 
    yml["results"] = results

    timestamp = datetime.datetime.now().strftime('%m%d%H%M%S')

    if args.save_results:
        file_name = f'results/seed_{args.seed}/{timestamp}.yaml'
        with open(file_name, 'w') as f:
            yaml.dump(yml, f)

def get_ids(test_id):
    path = f'../data/Hazumi_features/Hazumi{hazumi_version}_features.pkl'
    _, _, _, _, _, _, _, TP, _, _, _, ids = pickle.load(open(path, 'rb'), encoding='utf-8')

    columns = ['E', 'A', 'C', 'N', 'O']
    _df = pd.DataFrame.from_dict(TP, orient='index', columns=columns)
    df = (_df - _df.mean() ) / _df.std(ddof=0)

    test_profile = profiling(args.pmode, test_id, _df)

    train_id = []
    for id in ids:
        if profiling(args.pmode, id, df) == test_profile:
            train_id.append(id)

    train_id, valid_id = train_test_split(train_id)

    return train_id, valid_id

    
def main():
    path = f'../data/Hazumi_features/Hazumi{hazumi_version}_features.pkl'
    _, _, _, _, _, _, _, _, _, _, _, ids = pickle.load(open(path, 'rb'), encoding='utf-8')
    for test_id in ids:
        train_id, valid_id = get_ids(test_id)
        train_dataset = HazumiDataset_torch(version=hazumi_version, ids=train_id, modal=args.modal, ss=args.ss, binary=args.binary, regression=args.regression)
        sscaler = train_dataset.get_sscaler()
        valid_dataset = HazumiDataset_torch(version=hazumi_version, ids=valid_id, sscaler=sscaler, modal=args.modal, ss=args.ss, binary=args.binary, regression=args.regression)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

        model = UnimodalFNN(input_dim=input_dim, output_dim=output_dim).to(device)

        if args.binary:
            criterion = nn.BCEWithLogitsLoss()
        elif args.regression:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss() 

        optimizer = optim.SGD(model.parameters(), lr=lr)
        # optimizer = optim.Adam(model.parameters(), lr=lr)

        train_loss_list, valid_loss_list = run(model, train_loader, valid_loader, criterion, optimizer)

        evaluation(model, sscaler, test_id, criterion)

        if args.plot_results:
            show_plot(train_loss_list, valid_loss_list)

        # timestamp = datetime.datetime.now().strftime('%m%d%H%M%S')
        # if args.save_model:
        #     if args.ss:
        #         torch.save(model.state_dict(), f"model/seed_{args.seed}/ss-{args.pmode}-{pgroup}-{args.modal}-{timestamp}.pth")
        #     else:
        #         torch.save(model.state_dict(), f"model/seed_{args.seed}/ts-{args.pmode}-{pgroup}-{args.modal}-{timestamp}.pth")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--modal', type=str, default="t")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--pmode', type=int, default=0)
    parser.add_argument('--ss', action='store_true', default=False)
    parser.add_argument('--binary', action='store_true', default=False)
    parser.add_argument('--regression', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--plot_results', action='store_true', default=False)
    args = parser.parse_args()

    # seedの固定
    fix_seed(args.seed)
    
    # results用のディレクトリ作成
    if args.save_results:
        os.makedirs(f"results/seed_{args.seed}", exist_ok=True)
    if args.save_model:
        os.makedirs(f"model/seed_{args.seed}", exist_ok=True)

    # ハイパラ設定
    batch_size = 128
    test_batch_size = 32
    num_epochs = 100
    lr = 0.01

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.binary or args.regression:
        output_dim = 1
    else:
        output_dim = 3
    
    if args.modal == 't':
        input_dim = 768
    elif args.modal == 'a':
        input_dim = 384
    elif args.modal == 'v':
        input_dim = 66

    # train_version = "2012"
    # test_version = "2010"

    # train_version = "1911"
    # test_version = "1902"

    hazumi_version = "1911"

    # pgroup_num_dict = {0: 1, 1: 2, 2: 2, 3: 4, 4:2, 5:2, 6:2, 7:2, 8:2}
    # if args.pmode <= 8:
    #     pgroup_num = pgroup_num_dict[args.pmode]
    # else:
    #     pgroup_num = args.pmode - 7

    results = {}

    main()
    
    save_results()