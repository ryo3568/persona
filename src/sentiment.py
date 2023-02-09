import numpy as np
import argparse
import os
import glob
import string
import random 
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import LSTMSentimentModel, GRUSentimentModel
from dataloader import HazumiDataset
import utils
from utils.EarlyStopping import EarlyStopping

import warnings 
warnings.simplefilter('ignore')

import wandb 



def get_train_valid_sampler(trainset):
    size = len(trainset) 
    idx = list(range(size)) 
    split = int(0.1*size) 
    if split == 0:
        split = 1
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_Hazumi_loaders(test_file, batch_size=32, n_cluster=4, num_workers=2, pin_memory=False):
    trainset = HazumiDataset(test_file, n_cluster)
    testset = HazumiDataset(test_file, n_cluster, train=False, scaler=trainset.scaler) 

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


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    Loss = [] 

    assert not train or optimizer!=None 
    if train:
        model.train() 
    else:
        model.eval() 
    for data in dataloader:
        if train:
            optimizer.zero_grad() 
        
        text, visual, audio, _, s_ternary =\
        [d.cuda() for d in data[:-1]] if torch.cuda.is_available() else data[:-1]

        data = torch.cat((text, visual, audio), dim=-1)
            
        
        pred = model(data)

        label = s_ternary.view(-1)
        pred = pred.view(-1, 3)

        loss = loss_function(pred, label)

        pred = torch.argmax(pred, dim=1)

        Loss.append(loss.item())

        if train:
            loss.backward()
            optimizer.step() 

    avg_loss = round(np.sum(Loss)/len(Loss), 4)

    pred = pred.squeeze().cpu() 
    label = label.squeeze().cpu()

    return avg_loss, pred, label


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--n_cluster', type=int, default=4)
     
    args = parser.parse_args()

    config = {
        "epochs": 500,
        "batch_size": 1,
        "pretrain": args.pretrain,
        "D_h1": 256, 
        "D_h2": 64, 
        "weight_decay": 1e-5,
        "adam_lr": 1e-5,
        "dropout": 0.6,
        "n_cluster": args.n_cluster
    }

    project_name = 'sentiment'
    group_name = utils.randomname(5)

    testfiles = utils.get_files()

    for testfile in tqdm(testfiles, position=0, leave=True):

        model = GRUSentimentModel(config)
        loss_function = nn.CrossEntropyLoss() 

        if args.wandb:
            wandb.init(project=project_name, group=group_name, config=config, name=testfile)  

        if torch.cuda.is_available():
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=config["adam_lr"], weight_decay=config["weight_decay"])

        train_loader, valid_loader, test_loader = get_Hazumi_loaders(testfile, batch_size=config["batch_size"], n_cluster=args.n_cluster) 

        best_loss, best_acc,  best_val_loss, best_param = None, None, None, None

        es = EarlyStopping(patience=10, verbose=1)

        for epoch in range(config["epochs"]):
            trn_loss, _, _= train_or_eval_model(model, loss_function, train_loader, epoch, optimizer, True)
            val_loss, _, _= train_or_eval_model(model, loss_function, valid_loader, epoch)
            tst_loss, tst_pred, tst_label = train_or_eval_model(model, loss_function, test_loader, epoch)


            if best_loss == None or best_val_loss > val_loss:
                best_loss = tst_loss

                best_acc = accuracy_score(tst_label, tst_pred)

                best_val_loss = val_loss

                if args.pretrain:
                    best_param = model.state_dict()
        
            if es(val_loss):
                break
            
            if args.wandb:
                wandb.log({
                    '_trn loss': trn_loss,
                    '_val loss': val_loss
                })

        if args.pretrain:
            torch.save(best_param, f'../data/model/{testfile}.pth')


        if args.wandb:
            wandb.log({
                'tst loss': best_loss,
                'acc': best_acc
            })            
                
            wandb.finish()