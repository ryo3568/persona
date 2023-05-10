import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
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
    print(split)
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
        
        text, visual, audio, ans =\
        [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]

        data = torch.cat((text, visual, audio), dim=-1)

        pred = model(data)

        # tp_binary = tp_binary.view(-1)
        # loss += loss_function(pred, tp_binary)

        loss = loss_function(pred, ans)

        Loss.append(loss.item())

        if train:
            loss.backward()
            optimizer.step() 

    avg_loss = round(np.sum(Loss)/len(Loss), 4)

    return avg_loss, pred, ans


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--early_stop_num', type=int, default=10)
    parser.add_argument('--valid_rate', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--version', type=str, default="all")
     
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    config = {
        "epochs": 500,
        "D_h1": 256, 
        "D_h2": 64, 
        "weight_decay": 1e-5,
        "adam_lr": 1e-4,
        "dropout": 0.6,
        "valid_rate": args.valid_rate,
        "batch_size": args.batch_size,
        "version": args.version
    }

    project_name = 'TP regression'
    group_name = utils.randomname(5)

    testfiles = utils.get_files(args.version)
    Trait = ['extr', 'agre', 'cons', 'neur', 'open']
    Pred = []
    Ans = []
    for i, testfile in enumerate(tqdm(testfiles, position=0, leave=True)):

        model = LSTMModel(config)
        loss_function = nn.MSELoss() 

        if args.wandb:
            wandb.init(project=project_name, group=group_name, config=config, name=testfile)

        if args.cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=config["adam_lr"], weight_decay=config["weight_decay"])

        train_loader, valid_loader, test_loader =\
            get_Hazumi_loaders(args.version, testfile, batch_size=config["batch_size"])

        best_loss, best_val_loss, best_pred, best_ans = None, None, None, None

        es = EarlyStopping(patience=config['early_stop_num'])

        for epoch in range(config["epochs"]):
            trn_loss, trn_pred, trn_ans = train_or_eval_model(model, loss_function, train_loader, optimizer, True)
            val_loss, val_pred, val_ans = train_or_eval_model(model, loss_function, valid_loader)
            tst_loss, tst_pred, tst_ans = train_or_eval_model(model, loss_function, test_loader)

            if best_loss == None or best_val_loss > val_loss:
                best_loss, best_pred, best_ans = tst_loss, tst_pred, tst_ans
                best_val_loss = val_loss

            if es(val_loss):
                break
                
            if args.wandb:
                wandb.log({
                    "_trn loss": trn_loss,
                    "_val loss": val_loss,
                })

        ans = best_ans.view(-1)
        pred = best_pred.view(-1)

        if args.wandb:
            wandb.log({
                'tst loss': best_loss,
                '0E_loss': torch.square(ans[0] - pred[0]),
                '1A_loss': torch.square(ans[1] - pred[1]),
                '2C_loss': torch.square(ans[2] - pred[2]),
                '3N_loss': torch.square(ans[3] - pred[3]),
                '4O_loss': torch.square(ans[4] - pred[4]),
            })

            wandb.finish()


