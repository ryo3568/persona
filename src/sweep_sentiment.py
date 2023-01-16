import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from model import LSTMSentimentModel
from dataloader import HazumiDataset_sweep
from utils.EarlyStopping import EarlyStopping


import warnings 
warnings.simplefilter('ignore')

import wandb 

def train_epoch(model, loader, optimizer, loss_function, train=False):
    if train:
        model.train() 
    else: 
        model.eval() 

    Acc = []
    Loss = 0

    for data in loader:
        if train:
            optimizer.zero_grad() 

        text, visual, audio, _, s_ternary =\
        [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]

        data = torch.cat((text, visual, audio), dim=-1)    

        pred = model(data) 

        pred = pred.view(-1, 3)
        label = s_ternary.view(-1)

        loss = loss_function(pred, label)

        Loss += loss 

        pred = torch.argmax(pred, dim=1)

        Acc.append(accuracy_score(label.cpu() , pred.cpu()))
        
        if train:
            loss.backward() 
            optimizer.step() 
    
    return Loss / len(loader), sum(Acc) / len(loader)


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config 

        dataset = HazumiDataset_sweep()

        batch_size = args.batch_size
        kf = KFold(n_splits=5, shuffle=True)

        loss_function = nn.CrossEntropyLoss()

        Loss = []

        for _, (train_index, valid_index) in enumerate(kf.split(dataset)):
            train_dataset = Subset(dataset, train_index) 
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True) 
            valid_dataset = Subset(dataset, valid_index) 
            valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False) 


            model = LSTMSentimentModel(config.D_h1, config.D_h2, config.dropout).cuda()
            optimizer = optim.Adam(model.parameters(), lr=config.adam_lr, weight_decay=config.weight_decay)

            best_loss = -1

            es = EarlyStopping(patience=10)

            for epoch in range(args.epochs):
                train_epoch(model, train_loader, optimizer, loss_function, train=True) 
                loss, acc = train_epoch(model, valid_loader, optimizer, loss_function)
                
                wandb.log({
                    'loss': loss, 
                    'acc': acc,
                    'epoch': epoch
                    })

                if loss < best_loss or best_loss == -1:
                    best_loss = loss
                
                if es(loss):
                    break
            
            Loss.append(best_loss)


        wandb.log({'best_loss': sum(Loss) / len(Loss)})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    
    # 追加
    parser.add_argument('--trial_size', type=int, default=100, help='number of trial')
     
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'best_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'D_h1': {
                'values': [128, 256, 512]
            },   
            'D_h2': {
                'values': [32, 64, 128]
            }, 
            'weight_decay': {
                'values': [1e-5, 1e-3, 1e-1]
            },
            'adam_lr': {
                'values': [1e-5, 1e-3, 1e-1]
            },
            'dropout': {
                'values': [0.2, 0.4, 0.6, 0.8]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='sentiment sweep')

    wandb.agent(sweep_id, train, count=args.trial_size)