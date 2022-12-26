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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import optuna
from model import LSTMMultitaskModel, FNNMultitaskModel, biLSTMMultitaskModel
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


def train_or_eval_model(model, ploss_function, sloss_function, dataloader, epoch, optimizer=None, train=False, loss_weight=None):
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
        
        text, visual, audio, persona, _, s_ternary =\
        [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]

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

def objective(trial):

    D_i = 1218

    # ハイパラチューニング対象
    D_h = int(trial.suggest_discrete_uniform("D_h", 100, 1000, 50))
    in_droprate = trial.suggest_discrete_uniform("in_droprate", 0.0, 0.5, 0.05)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-10, 1e-3)
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
    loss_weight = trial.suggest_discrete_uniform('alpha', 0.05, 0.95, 0.05)

    config = dict(trial.params) 
    config['trial.number'] = trial.number

    config.update(vars(args))

    testfiles = []
    for f in glob.glob('../data/Hazumi1911/dumpfiles/*.csv'):
        testfiles.append(os.path.splitext(os.path.basename(f))[0])

    testfiles = sorted(testfiles)

    loss = []

    Trait = ['extr', 'agre', 'cons', 'neur', 'open']

    for testfile in tqdm(testfiles, position=0, leave=True):

        model = LSTMMultitaskModel(D_i, D_h, prediction=False)
        pLoss_function = nn.BCELoss() # 性格特性
        sLoss_function = nn.CrossEntropyLoss() # 心象

        Acc = dict.fromkeys(Trait)

        wandb.init(project='multitask', group=group_name, config=config, name=testfile)

                    
        if args.cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)

        train_loader, valid_loader, test_loader = get_Hazumi_loaders(testfile, batch_size=args.batch_size, valid=0.1) 

        best_val_loss, best_loss, best_ploss, best_sloss = None, None, None, None

        es = EarlyStopping(patience=10, verbose=1)

        for epoch in range(args.epochs):
            trn_loss, trn_ploss, trn_sloss, _, _, _, _ = train_or_eval_model(model, pLoss_function, sLoss_function, train_loader, epoch, optimizer, True, loss_weight=loss_weight)
            val_loss, val_ploss, val_sloss, _, _, _, _ = train_or_eval_model(model, pLoss_function, sLoss_function, valid_loader, epoch, loss_weight=loss_weight)
            tst_loss, tst_ploss, tst_sloss, tst_ppred, tst_spred, tst_plabel, tst_slabel = train_or_eval_model(model, pLoss_function, sLoss_function, test_loader, epoch, loss_weight=loss_weight)


            if best_loss == None or best_val_loss > val_ploss:
                best_loss, best_ploss, best_sloss = tst_loss, tst_ploss, tst_sloss

                best_pacc = accuracy_score(tst_plabel, tst_ppred > 0.5)

                best_sacc = accuracy_score(tst_slabel, tst_spred)

                for i, trait in enumerate(Trait):
                    Acc[trait] = accuracy_score([tst_plabel[i]], [tst_ppred[i] > 0.5])

                best_val_loss = val_ploss
            
            if es(val_ploss):
                break

            wandb.log({
                "_trn loss": trn_loss,
                "_trn ploss": trn_ploss, 
                "_trn sloss": trn_sloss,
                "_val loss": val_loss,
                "_val ploss": val_ploss, 
                "_val sloss": val_sloss,
            })
            
        loss.append(best_ploss) # best_ploss or best_loss

        wandb.log({
            'tst loss': best_loss,
            'tst ploss': best_ploss, 
            'tst sloss': best_sloss, 
            'tst pacc': best_pacc,
            'tst sacc': best_sacc,
            'extr': Acc['extr'],
            'agre': Acc['agre'],
            'cons': Acc['cons'],
            'neur': Acc['neur'],
            'open': Acc['open']
        })

        wandb.finish()

    return np.array(loss).mean()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
 
    # 追加
    parser.add_argument('--trail_size', type=int, default=1, help='number of trail')
     
    args = parser.parse_args()


    args.cuda = torch.cuda.is_available() and not args.no_cuda 
    if args.cuda:
        print('Running on GPU') 
    else:
        print('Running on CPU')

    group_name = randomname(5)

    study = optuna.create_study() 
    study.optimize(objective, n_trials=args.trail_size)


    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))