import numpy as np
import os
import glob
import argparse
import random 
import string
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import LSTMMultitaskModel, GRUMultitaskModel, RNNMultitaskModel
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


def train_or_eval_model(model, ploss_function, sloss_function, dataloader, optimizer=None, train=False, loss_weight=None):
    pLoss = []
    sLoss = []
    Loss = [] 

    assert not train or optimizer!=None 
    if train:
        model.train() 
    else:
        model.eval() 
    for data in dataloader:
        if train:
            optimizer.zero_grad() 
        
        text, visual, audio, persona, s_ternary =\
        [d.cuda() for d in data[:-1]] if torch.cuda.is_available()  else data[:-1]

        # data = visual
        # data = torch.cat((visual, text), dim=-1)
        data = torch.cat((text, visual, audio), dim=-1)            
        
        ppred, spred = model(data)

        slabel = s_ternary.view(-1)
        spred = spred.view(-1, 3)

        plabel = persona

        # persona = persona.repeat(1, data.shape[1], 1)

        ploss = ploss_function(ppred, plabel)

        sloss = sloss_function(spred, slabel)

        
        loss = loss_weight * ploss + (1-loss_weight) * sloss

        spred = torch.argmax(spred, dim=1)

        pLoss.append(ploss.item())
        sLoss.append(sloss.item())
        Loss.append(loss.item())


        if train:
            loss.backward()
            optimizer.step() 

    avg_loss = round(np.sum(Loss)/len(Loss), 4)
    avg_ploss = round(np.sum(pLoss)/len(pLoss), 4)
    avg_sloss = round(np.sum(sLoss)/len(sLoss), 4)

    ppred = ppred.squeeze().cpu()
    spred = spred.squeeze().cpu()
    plabel = plabel.squeeze().cpu()
    slabel = slabel.squeeze().cpu()

    return avg_loss, avg_ploss, avg_sloss, ppred, spred, plabel, slabel



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=0, help='0:RNN, 1:LSTM, 2:GRU')
     
    args = parser.parse_args()

    config = {
        "epochs": 500,
        "batch_size": 1,
        "D_h1": 256, 
        "D_h2": 64, 
        "weight_decay": 1e-5,
        "adam_lr": 1e-5,
        "dropout": 0.6,
        "loss_weight": 0.6
    }

    project_name = 'multitask'
    group_name = randomname(5)

    testfiles = get_files()
    Trait = ['extr', 'agre', 'cons', 'neur', 'open']

    for testfile in tqdm(testfiles, position=0, leave=True):

        if args.model == 0:
            model = RNNMultitaskModel(config["D_h1"], config["D_h2"], config["dropout"])
        elif args.model ==1:
            model = LSTMMultitaskModel(config["D_h1"], config["D_h2"], config["dropout"])
        else:
            model = GRUMultitaskModel(config["D_h1"], config["D_h2"], config["dropout"])

        # model = GRUMultitaskModel(config["D_h1"], config["D_h2"], config["dropout"])
        pLoss_function = nn.BCELoss() # 性格特性
        sLoss_function = nn.CrossEntropyLoss() # 心象

        Acc = dict.fromkeys(Trait)

        if args.model == 0:
            notes = 'RNN'
        elif args.model == 1:
            notes = 'LSTM'
        else:
            notes = 'GRU'

        wandb.init(project=project_name, group=group_name, config=config, name=testfile, notes=notes)

        if torch.cuda.is_available():
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=config["adam_lr"], weight_decay=config["weight_decay"])

        train_loader, valid_loader, test_loader = get_Hazumi_loaders(testfile, batch_size=config["batch_size"], valid=0.1) 

        best_val_loss, best_loss, best_ploss, best_sloss = None, None, None, None

        es = EarlyStopping(patience=10, verbose=1)

        for epoch in range(config["epochs"]):
            trn_loss, trn_ploss, trn_sloss, _, _, _, _ = train_or_eval_model(model, pLoss_function, sLoss_function, train_loader, optimizer, True, loss_weight=config["loss_weight"])
            val_loss, val_ploss, val_sloss, _, _, _, _ = train_or_eval_model(model, pLoss_function, sLoss_function, valid_loader, loss_weight=config["loss_weight"])
            tst_loss, tst_ploss, tst_sloss, tst_ppred, tst_spred, tst_plabel, tst_slabel = train_or_eval_model(model, pLoss_function, sLoss_function, test_loader, loss_weight=config["loss_weight"])


            if best_loss == None or best_val_loss > val_ploss:
                best_loss, best_ploss, best_sloss = tst_loss, tst_ploss, tst_sloss

                best_pacc = accuracy_score(tst_plabel, tst_ppred > 0.5)

                best_sacc = accuracy_score(tst_slabel, tst_spred)

                for i, trait in enumerate(Trait):
                    Acc[trait] = accuracy_score([tst_plabel[i]], [tst_ppred[i] > 0.5])

                best_val_loss = val_ploss
            
            if es(val_loss):
                break

            wandb.log({
                "_trn loss": trn_loss,
                "_trn ploss": trn_ploss, 
                "_trn sloss": trn_sloss,
                "_val loss": val_loss,
                "_val ploss": val_ploss, 
                "_val sloss": val_sloss,
            })

        wandb.log({
            'tst loss': best_loss,
            'tst ploss': best_ploss, 
            'tst sloss': best_sloss, 
            'tst pacc': best_pacc,
            'tst sacc': best_sacc,
            '1extr': Acc['extr'],
            '2agre': Acc['agre'],
            '3cons': Acc['cons'],
            '4neur': Acc['neur'],
            '5open': Acc['open']
        })

        wandb.finish()