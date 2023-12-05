import pickle
import argparse
import yaml
import datetime
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn 
import torch.optim as optim 

from dataset import HazumiDataset_multi, HazumiTestDataset_multi
from model import UnimodalFNN, LatefusionFNN
from utils.EarlyStopping import EarlyStopping
from utils import fix_seed, profiling


def train_fn(model, train_loader, criterion, optimizer):

    # 1 epoch training 
    train_loss = 0.0 
    num_train = 0 

    # model 学習モードに設定
    model.train() 

    for t_x, a_x, v_x, y in train_loader:
        # batch数の累積
        num_train += len(y)

        t_x, a_x, v_x, y = t_x.to(device), a_x.to(device), v_x.to(device), y.view(-1).to(device)
        # 勾配をリセット
        optimizer.zero_grad() 
        # 推論
        output = model(t_x, a_x, v_x)
        # lossの計算
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
        for t_x, a_x, v_x, y in test_loader:
            num_valid += len(y)
            # x, y = x.to(device), y.view(-1).to(device) 
            t_x, a_x, v_x, y = t_x.to(device), a_x.to(device), v_x.to(device), y.view(-1).to(device)
            output = model(t_x, a_x, v_x)
            loss = criterion(output, y)
            valid_loss += loss.item()
            # outputの蓄積
            _output = torch.argmax(output, dim=1)
            outputs.extend(_output.tolist())
            # labelの蓄積
            labels.extend(y.tolist())

        valid_loss = valid_loss / num_valid

        acc = accuracy_score(labels, outputs)

    return valid_loss, acc


def run(model, train_loader, valid_loader, criterion, optimizer):

    train_loss_list = [] 
    valid_loss_list = [] 

    es = EarlyStopping(patience=10)

    # for epoch in range(num_epochs):
    epoch = 1
    while True:

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


def evaluation(model, a_scaler, v_scaler, test_id):
    for id in test_id:
        test_dataset = HazumiTestDataset_multi(version=test_version, id=id, a_scaler=a_scaler, v_scaler=v_scaler, ss=args.ss)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

        # 評価用のコード
        num_test = 0 
        outputs = []
        labels = []

        # model　評価モードに設定
        model.eval() 

        # 評価の際に勾配を計算しないようにする
        with torch.no_grad():
            for t_x, a_x, v_x, y in test_loader:
                num_test += len(y)
                t_x, a_x, v_x, y = t_x.to(device), a_x.to(device), v_x.to(device), y.to(device)
                output = model(t_x, a_x, v_x)
                # outputの蓄積
                _output = torch.argmax(output, dim=1)
                outputs.extend(_output.tolist())
                # labelの蓄積
                labels.extend(y.tolist())

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
    
    yml = {}
    yml["config"] = config 
    yml["results"] = results

    timestamp = datetime.datetime.now().strftime('%m%d%H%M%S')

    if args.results_save:
        with open(f"results/{timestamp}.yaml", 'w') as f:
            yaml.dump(yml, f)

def get_ids(pgroup):
    path = f'../data/Hazumi_features/Hazumi{train_version}_features.pkl'
    _, _, _, _, _, _, _, _, _, _, _, _train_id = pickle.load(open(path, 'rb'), encoding='utf-8')

    train_id = []
    for id in _train_id:
        if profiling(args.pmode, id) == pgroup:
            train_id.append(id)

    train_id, valid_id = train_test_split(train_id)

    path = f'../data/Hazumi_features/Hazumi{test_version}_features.pkl'
    _, _, _, _, _, _, _, _, _, _, _, _test_id = pickle.load(open(path, 'rb'), encoding='utf-8')

    test_id = [] 
    for id in _test_id:
        if profiling(args.pmode, id) == pgroup:
            test_id.append(id)
    
    return train_id, valid_id, test_id

    
def main(pgroup=0):
    train_id, valid_id, test_id = get_ids(pgroup)
    print(f"pgroup : {pgroup}")
    train_dataset = HazumiDataset_multi(version=train_version, ids=train_id, ss=args.ss)
    a_scaler, v_scaler = train_dataset.get_sscaler()
    valid_dataset = HazumiDataset_multi(version=train_version, ids=valid_id, a_scaler=a_scaler, v_scaler=v_scaler, ss=args.ss)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)


    model = LatefusionFNN(input_dim=input_dim, num_classes=num_classes, ss=args.ss, pmode=args.pmode, pgroup=pgroup, modal=args.modal).to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_loss_list, valid_loss_list = run(model, train_loader, valid_loader, criterion, optimizer)

    evaluation(model, a_scaler, v_scaler, test_id)

    if args.results_plot:
        show_plot(train_loss_list, valid_loss_list)

    timestamp = datetime.datetime.now().strftime('%m%d%H%M%S')
    if args.model_save:
        if args.ss:
            torch.save(model.state_dict(), f"model/ss-{args.pmode}-{pgroup}-{args.modal}-{timestamp}.pth")
        else:
            torch.save(model.state_dict(), f"model/ts-{args.pmode}-{pgroup}-{args.modal}-{timestamp}.pth")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--modal', type=str, default="tav")
    parser.add_argument('--pmode', type=int, default=0)
    parser.add_argument('--ss', action='store_true', default=False)
    parser.add_argument('--binary', action='store_true', default=False)
    parser.add_argument('--results_save', action='store_true', default=False)
    parser.add_argument('--model_save', action='store_true', default=False)
    parser.add_argument('--results_plot', action='store_true', default=False)
    args = parser.parse_args()

    # seedの固定
    fix_seed()

    # ハイパラ設定
    batch_size = 128 
    test_batch_size = 32
    # num_epochs = 500
    lr = 0.01

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 3

    input_dim = 0 
    if 't' in args.modal:
        input_dim += 32
    if 'a' in args.modal:
        input_dim += 32
    if 'v' in args.modal:
        input_dim += 32

    train_version = "1911"
    test_version = "1902"

    pgroup_num = {0: 1, 1: 2, 2: 2, 3: 4}

    results = {}

    for pgroup in range(pgroup_num[args.pmode]):
        main(pgroup)
    
    save_results()