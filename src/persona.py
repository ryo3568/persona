import numpy as np
import argparse
import os
import glob
import random 
import string
from tqdm import tqdm
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import optuna
from model import LSTMModel, FNNModel
from dataloader import HazumiDataset
from utils.EarlyStopping import EarlyStopping

import warnings 
warnings.simplefilter('ignore')

import wandb 

def randomname(n):
   return ''.join(random.choices(string.ascii_letters + string.digits, k=n))


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
    losses = []
    preds = [] 
    labels = []
    separate_loss = []
    assert not train or optimizer!=None 
    if train:
        model.train() 
    else:
        model.eval() 
    for i, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad() 
        
        text, visual, audio, persona, sentiment, s_ternary =\
        [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]

        # data = audio
        # data = torch.cat((visual, audio), dim=-1)
        data = torch.cat((text, visual, audio), dim=-1)
            
        
        pred = model(data)


        # Model = Netの場合は有効にする
        # persona = persona.repeat(1, data.shape[1], 1) 

        loss = loss_function(pred, persona)
        
        if not train:
            for i in range(5):
                tmp_loss = round(loss_function(pred[:, i], persona[:, i]).item(), 3)
                separate_loss.append(tmp_loss)
        

        # 学習ログ
        losses.append(loss.item())
        preds.append(pred.data.cpu().numpy())
        labels.append(persona.data.cpu().numpy())


        if train:
            loss.backward()
            optimizer.step() 


    avg_loss = round(np.sum(losses)/len(losses), 4)

    return avg_loss, preds, labels, separate_loss

def objective(trial):

    D_i = 1218 

    # ハイパラチューニング対象
    D_h = int(trial.suggest_discrete_uniform("D_h", 50, 300, 50))
    D_o = int(trial.suggest_discrete_uniform("D_o", 10, 32, 2))
    # in_droprate = trial.suggest_discrete_uniform("in_droprate", 0.0, 0.2, 0.05)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-10, 1e-3)
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)

    config = dict(trial.params) 
    config['D_i'] = D_i
    config['trial.number'] = trial.number

    config.update(vars(args))

    testfiles = []
    for f in glob.glob('../data/Hazumi1911/dumpfiles/*.csv'):
        testfiles.append(os.path.splitext(os.path.basename(f))[0])

    testfiles = sorted(testfiles)


    persona_loss = []

    extr_loss = [] 
    agre_loss = [] 
    cons_loss = [] 
    neur_loss = [] 
    open_loss = [] 
    

    project_name = randomname(10)


    for testfile in tqdm(testfiles, position=0, leave=True):

        model = LSTMModel(D_i, D_h, n_classes=5)
        loss_function = nn.MSELoss() # 性格特性

        wandb.init(project=project_name, config=config, name=testfile)

        if args.cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)

        train_loader, valid_loader, test_loader = get_Hazumi_loaders(testfile, batch_size=args.batch_size, valid=0.1) 

        best_loss, best_val_loss = None, None

        es = EarlyStopping(patience=10)

        for epoch in range(args.epochs):
            trn_loss, _, _, _= train_or_eval_model(model, loss_function, train_loader, epoch, optimizer, True)
            val_loss, _, _, _ = train_or_eval_model(model, loss_function, valid_loader, epoch)
            tst_loss, _, _, tst_sep_loss = train_or_eval_model(model, loss_function, test_loader, epoch)


            if best_loss == None or best_val_loss > val_loss:
                best_loss, best_sep_loss = tst_loss, tst_sep_loss

                best_val_loss = val_loss

            if es(val_loss):
                break

            wandb.log({
                "_train loss": trn_loss,
                "_valid loss": val_loss,
                "_test loss": tst_loss,
                "extraversion": tst_sep_loss[0],
                "agreeableness": tst_sep_loss[1],
                "conscientiousness": tst_sep_loss[2],
                "neuroticism": tst_sep_loss[3],
                "openness": tst_sep_loss[4],
            })

        wandb.log({
            'best loss': best_loss,
            'best etra loss': best_sep_loss[0],
            'best agre loss': best_sep_loss[1],
            'best cons loss': best_sep_loss[2], 
            'best neur loss': best_sep_loss[3], 
            'best open loss': best_sep_loss[4]
        })
        

        persona_loss.append(best_loss)

        extr_loss.append(best_sep_loss[0])
        agre_loss.append(best_sep_loss[1])
        cons_loss.append(best_sep_loss[2])
        neur_loss.append(best_sep_loss[3]) 
        open_loss.append(best_sep_loss[4]) 

        wandb.finish()
        

    wandb.init(project=project_name, config=args, name='stats')
    wandb.log({
    '0best loss': np.array(persona_loss).mean(),
    '1best etra loss': np.array(extr_loss).mean(),
    '2best agre loss': np.array(agre_loss).mean(),
    '3best cons loss': np.array(cons_loss).mean(), 
    '4best neur loss': np.array(neur_loss).mean(), 
    '5best open loss': np.array(open_loss).mean()
    })
    wandb.finish()

    
    return np.array(persona_loss).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--batch-size', type=int, default=5, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    
    # 追加
    parser.add_argument('--trail_size', type=int, default=1, help='number of trail')
     
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available() and not args.no_cuda 
    if args.cuda:
        print('Running on GPU') 
    else:
        print('Running on CPU')

    study = optuna.create_study() 
    study.optimize(objective, n_trials=args.trail_size)

    print(study.best_params)
