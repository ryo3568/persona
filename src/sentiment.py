import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from model import LateFusionModel, TextModel, AudioModel, VisualModel
from dataloader import HazumiDataset
import utils
from utils.EarlyStopping import EarlyStopping

import warnings 
warnings.simplefilter('ignore')

import wandb 


def get_train_valid_sampler(trainset):
    size = len(trainset) 
    idx = list(range(size)) 
    split = int(0.2*size) 
    np.random.shuffle(idx)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_Hazumi_loaders(version, test_file, batch_size, num_workers=2, pin_memory=False):
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
        
        text, visual, audio, _, _, _, ts =\
        [d.cuda() for d in data[:-1]] if torch.cuda.is_available() else data[:-1]

        # data = torch.cat(text, audio, visual, dim=-1)

        if config["modal"] == 't':
            pred, _ = model(text)
        elif config["modal"] == 'a':
            pred, _ = model(audio) 
        elif config["modal"] == 'v':
            pred, _ = model(visual)
        else:
            pred = model(text, audio, visual)

        label = ts.view(-1)
        pred = pred.view(-1, 3)

        loss = loss_function(pred, label)

        if train:
            loss.backward()
            optimizer.step() 

        pred = torch.argmax(pred, dim=1)

        Loss.append(loss.item())

    avg_loss = round(np.sum(Loss)/len(Loss), 4)
    pred = pred.squeeze().cpu() 
    label = label.squeeze().cpu()

    return avg_loss, pred, label


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--early_stop_num', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--version', type=str, default="1911")
    parser.add_argument('--modal', type=str, default="tav")

     
    args = parser.parse_args()

    config = {
        "epochs": 500,
        "D_h1": 256, 
        "D_h2": 64, 
        "weight_decay": 1e-5,
        "adam_lr": 1e-5,
        "dropout": 0.6,
        "early_stop_num": args.early_stop_num,
        "batch_size": args.batch_size,
        "version": args.version,
        "modal": args.modal
    }

    project_name = 'third sentiment'
    group_name = utils.randomname(5)

    testfiles = utils.get_files(args.version)

    for testfile in tqdm(testfiles, position=0, leave=True):

        if 't' == config["modal"]:
            model = TextModel(config)
            model_path = f"../data/model/text/{testfile}"
        elif 'a' == config["modal"]:
            model = AudioModel(config)
            model_path = f"../data/model/audio/{testfile}"
        elif 'v' == config["modal"]:
            model = VisualModel(config)
            model_path = f"../data/model/visual/{testfile}"
        else:
            model = LateFusionModel(config, testfile)
            
        loss_function = nn.CrossEntropyLoss() 

        if args.wandb:
            wandb.init(project=project_name, group=group_name, config=config, name=testfile)  

        if torch.cuda.is_available():
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=config["adam_lr"], weight_decay=config["weight_decay"])

        train_loader, valid_loader, test_loader =\
            get_Hazumi_loaders(args.version, testfile, batch_size=config["batch_size"]) 

        best_loss, best_acc,  best_val_loss, best_param = None, None, None, None

        es = EarlyStopping(patience=config['early_stop_num'])

        for epoch in range(config["epochs"]):
            trn_loss, _, _= train_or_eval_model(model, loss_function, train_loader, optimizer, True)
            val_loss, _, _= train_or_eval_model(model, loss_function, valid_loader)
            tst_loss, tst_pred, tst_label = train_or_eval_model(model, loss_function, test_loader)


            if best_loss == None or best_val_loss > val_loss:
                best_loss = tst_loss

                best_acc = accuracy_score(tst_label, tst_pred)

                best_val_loss = val_loss

            if es(val_loss):
                break
            
            if args.wandb:
                wandb.log({
                    '_trn loss': trn_loss,
                    '_val loss': val_loss
                })

        if args.wandb:
            wandb.log({
                'tst loss': best_loss,
                'acc': best_acc
            })            
                
            wandb.finish()

        torch.save(model.state_dict(), model_path)