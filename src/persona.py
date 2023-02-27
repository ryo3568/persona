import numpy as np
import argparse
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import  accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import LSTMModel
from dataloader import HazumiDataset
import utils
from utils.EarlyStopping import EarlyStopping

import warnings 
warnings.simplefilter('ignore')

import wandb 

def get_train_valid_sampler(trainset):
    size = len(trainset) 
    idx = list(range(size)) 
    split = int(args.valid_rate*size) 
    np.random.shuffle(idx)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_Hazumi_loaders(test_file, batch_size=1, window_size=-1, step_size=1, num_workers=2, pin_memory=False):
    trainset = HazumiDataset(test_file, window_size, step_size)
    testset = HazumiDataset(test_file, window_size, step_size, train=False, scaler=trainset.scaler) 

    train_sampler, valid_sampler = get_train_valid_sampler(trainset)

    train_loader = DataLoader(trainset, 
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader 


def train_or_eval_model(model, loss_function, dataloader, optimizer=None, train=False):
    Loss = []
    Pred = []
    Label = []

    assert not train or optimizer!=None 
    if train:
        model.train() 
    else:
        model.eval() 

    for data in dataloader:
        if train:
            optimizer.zero_grad() 
        
        text, visual, audio, tp_binary, _ =\
        [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]

        data = torch.cat((text, visual, audio), dim=-1)

        # seq_len = data.shape[1]
        # start_index = random.randint(0, seq_len - args.window_size)

        # data = data[:, start_index: start_index+args.window_size, :]

        data = utils.rolling_window(data, args.window_size, args.step_size)

        loss = 0
        preds = []

        for d in data:
            pred = model(d)
            loss += loss_function(pred, tp_binary)
            preds.append(pred)
        
        preds = torch.concat(preds)
        preds = torch.mean(preds, dim=0)

        loss /= len(data)
        Pred.append(preds.tolist())
        Label.append(tp_binary.tolist())
        Loss.append(loss.item())

        if train:
            loss.backward()
            optimizer.step() 

    avg_loss = round(np.sum(Loss)/len(Loss), 4)

    Pred = list(utils.flatten(Pred))
    Pred = [1 if x >= 0.5 else 0 for x in Pred]
    Label = list(utils.flatten(Label))

    return avg_loss, Pred, Label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--early_stop_num', type=int, default=10, metavar='BS', help='batch size')
    parser.add_argument('--window_size', type=int, default=-1)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--valid_rate', type=float, default=0.2)

     
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()


    config = {
        "epochs": 500,
        "batch_size": 1,
        "finetune": args.finetune,
        "D_h1": 256, 
        "D_h2": 64, 
        "weight_decay": 1e-5,
        "adam_lr": 1e-5,
        "dropout": 0.6,
        "early_stop_num": args.early_stop_num,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "valid_rate": args.valid_rate
    }

    project_name = 'persona'
    group_name = utils.randomname(5)

    testfiles = utils.get_files()
    Trait = ['extr', 'agre', 'cons', 'neur', 'open']
    Pred = []
    Label = []
    for i, testfile in enumerate(tqdm(testfiles, position=0, leave=True)):

        model = LSTMModel(config)

        if args.finetune:
            model.load_state_dict(torch.load(f'../data/model/{testfile}.pth'), strict=False) 

        loss_function = nn.BCELoss() 

        Acc = dict.fromkeys(Trait)

        if args.wandb:
            wandb.init(project=project_name, group=group_name, config=config, name=testfile)

        if args.cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=config["adam_lr"], weight_decay=config["weight_decay"])

        train_loader, valid_loader, test_loader =\
            get_Hazumi_loaders(testfile, batch_size=config["batch_size"], 
                               window_size=args.window_size, 
                               step_size=args.step_size) 

        best_loss, best_val_loss, best_pred, best_label = None, None, None, None

        es = EarlyStopping(patience=config['early_stop_num'])

        for epoch in range(config["epochs"]):
            trn_loss, trn_pred, trn_label = train_or_eval_model(model, loss_function, train_loader, optimizer, True)
            val_loss, val_pred, val_label = train_or_eval_model(model, loss_function, valid_loader)
            tst_loss, tst_pred, tst_label = train_or_eval_model(model, loss_function, test_loader)


            if best_loss == None or best_val_loss > val_loss:
                best_loss, best_label, best_pred= tst_loss, tst_label, tst_pred

                best_acc = accuracy_score(tst_label, tst_pred)

                for j, trait in enumerate(Trait):
                    Acc[trait] = accuracy_score([tst_label[j]], [tst_pred[j]])


                best_val_loss = val_loss

            
            trn_acc = accuracy_score(trn_label, trn_pred)
            val_acc = accuracy_score(val_label, val_pred)


            if es(val_loss):
                break
                
            if args.wandb:
                wandb.log({
                    "_trn loss": trn_loss,
                    "_val loss": val_loss,
                    "trn acc": trn_acc,
                    'val acc': val_acc
                })

        if args.wandb:
            wandb.log({
                'tst loss': best_loss,
                'acc': best_acc,
                '1extr': Acc['extr'],
                '2agre': Acc['agre'],
                '3cons': Acc['cons'],
                '4neur': Acc['neur'],
                '5open': Acc['open']
            })

            wandb.finish()

        Pred.append(best_pred)
        Label.append(best_label)

    utils.calc_confusion(Pred, Label)


