import numpy as np
import argparse
import os
import glob 
import random 
import string
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import  accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import LSTMModel
from dataloader import HazumiDataset
from utils.EarlyStopping import EarlyStopping

import warnings 
warnings.simplefilter('ignore')

import wandb 

def randomname(n):
   return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def get_files():
    testfiles = []
    for f in glob.glob('../data/Hazumi1911/dumpfiles/*.csv'):
        testfiles.append(os.path.splitext(os.path.basename(f))[0])

    testfiles = sorted(testfiles)
    return testfiles

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset) 
    idx = list(range(size)) 
    split = int(valid*size) 
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_Hazumi_loaders(test_file, batch_size=32, valid=0.1, num_workers=2, pin_memory=False):
    trainset = HazumiDataset(test_file)
    testset = HazumiDataset(test_file, train=False, scaler=trainset.scaler) 

    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

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
        
        text, visual, audio, persona, _, =\
        [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]

        data = torch.cat((text, visual, audio), dim=-1)
            
        
        pred = model(data)

        loss = loss_function(pred, persona)
        
        Loss.append(loss.item())

        if train:
            loss.backward()
            optimizer.step() 


    avg_loss = round(np.sum(Loss)/len(Loss), 4)

    pred = pred.squeeze().cpu() 
    label = persona.squeeze().cpu()

    return avg_loss, pred, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune', action='store_true', default=False)

     
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
        "dropout": 0.6
    }

    project_name = 'persona'
    group_name = randomname(5)

    testfiles = get_files()
    Trait = ['extr', 'agre', 'cons', 'neur', 'open']

    for testfile in tqdm(testfiles, position=0, leave=True):

        model = LSTMModel(config["D_h1"], config["D_h2"], config["dropout"])

        if args.finetune:
            model.load_state_dict(torch.load(f'../data/model/{testfile}.pth'), strict=False) 

        loss_function = nn.BCELoss() 

        Acc = dict.fromkeys(Trait)

        wandb.init(project=project_name, group=group_name, config=config, name=testfile)

        if args.cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=config["adam_lr"], weight_decay=config["weight_decay"])

        train_loader, valid_loader, test_loader = get_Hazumi_loaders(testfile, batch_size=config["batch_size"], valid=0.1) 

        best_loss, best_val_loss = None, None

        es = EarlyStopping(patience=10)

        for epoch in range(config["epochs"]):
            trn_loss, _, _ = train_or_eval_model(model, loss_function, train_loader, epoch, optimizer, True)
            val_loss, _, _ = train_or_eval_model(model, loss_function, valid_loader, epoch)
            tst_loss, tst_pred, tst_label = train_or_eval_model(model, loss_function, test_loader, epoch)


            if best_loss == None or best_val_loss > val_loss:
                best_loss = tst_loss

                best_acc = accuracy_score(tst_label, tst_pred > 0.5)

                for i, trait in enumerate(Trait):
                    Acc[trait] = accuracy_score([tst_label[i]], [tst_pred[i] > 0.5])

                best_val_loss = val_loss

            if es(val_loss):
                break

            wandb.log({
                "_trn loss": trn_loss,
                "_val loss": val_loss,
            })


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

