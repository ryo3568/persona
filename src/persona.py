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

def get_Hazumi_loaders(version, test_file, batch_size=1, num_workers=2, pin_memory=False):
    trainset = HazumiDataset(version, test_file)
    testset = HazumiDataset(version, test_file, train=False, scaler=trainset.scaler) 

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
    assert not train or optimizer!=None 
    if train:
        model.train() 
    else:
        model.eval() 

    for data in dataloader:
        if train:
            optimizer.zero_grad() 
        
        text, visual, audio, tp_binary =\
        [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]

        data = torch.cat((text, visual, audio), dim=-1)

        data = data[:, :50, :]

        loss = 0
        pred = model(data)

        tp_binary = tp_binary.view(-1)
        loss += loss_function(pred, tp_binary)

        Loss.append(loss.item())

        if train:
            loss.backward()
            optimizer.step() 

    avg_loss = round(np.sum(Loss)/len(Loss), 4)

    return avg_loss, pred, tp_binary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--early_stop_num', type=int, default=10)
    parser.add_argument('--valid_rate', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--version', type=str, default="all")
     
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    config = {
        "epochs": 500,
        "batch_size": args.batch_size,
        "finetune": args.finetune,
        "D_h1": 256, 
        "D_h2": 64, 
        "weight_decay": 1e-5,
        "adam_lr": 1e-5,
        "dropout": 0.6,
        "early_stop_num": args.early_stop_num,
        "valid_rate": args.valid_rate,
        "version": args.version
    }

    project_name = 'persona'
    group_name = utils.randomname(5)

    testfiles = utils.get_files(args.version)
    Trait = ['extr', 'agre', 'cons', 'neur', 'open']
    Pred = []
    Label = []
    for i, testfile in enumerate(tqdm(testfiles, position=0, leave=True)):
        if i == 26:
            break
        model = LSTMModel(config)
        if args.finetune:
            model.load_state_dict(torch.load(f'../data/model/{testfile}.pth'), strict=False) 

        loss_function = nn.CrossEntropyLoss() 

        Acc = dict.fromkeys(Trait)

        if args.wandb:
            wandb.init(project=project_name, group=group_name, config=config, name=testfile)

        if args.cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=config["adam_lr"], weight_decay=config["weight_decay"])

        train_loader, valid_loader, test_loader =\
            get_Hazumi_loaders(args.version, testfile, batch_size=config["batch_size"])

        best_loss, best_val_loss, best_pred, best_label = None, None, None, None

        es = EarlyStopping(patience=config['early_stop_num'])

        for epoch in range(config["epochs"]):
            trn_loss, trn_pred, trn_label = train_or_eval_model(model, loss_function, train_loader, optimizer, True)
            val_loss, val_pred, val_label = train_or_eval_model(model, loss_function, valid_loader)
            tst_loss, tst_pred, tst_label = train_or_eval_model(model, loss_function, test_loader)


            if best_loss == None or best_val_loss > val_loss:
                best_loss, best_label, best_pred= tst_loss, tst_label, tst_pred
                best_val_loss = val_loss

            if es(val_loss):
                acc = 1 if torch.argmax(tst_pred, dim=1).item() == tst_label.item() else 0
                break
                
            if args.wandb:
                wandb.log({
                    "_trn loss": trn_loss,
                    "_val loss": val_loss,
                })

        if args.wandb:
            wandb.log({
                'tst loss': best_loss,
                'acc': acc,
            })

            wandb.finish()


