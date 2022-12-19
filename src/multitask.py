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
import mlflow 
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


def train_or_eval_model(model, loss_function1, loss_function2, dataloader, epoch, optimizer=None, train=False, loss_weight=None):
    persona_losses = []
    sentiment_losses = []
    all_losses = [] 
    pred_personas = []
    pred_sentiments = [] 
    y_personas = []
    y_sentiments = []
    separate_loss = []
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
        
        pred_persona, pred_sentiment = model(data)


        s_ternary = s_ternary.view(-1)
        y_sentiment = s_ternary
        pred_sentiment = pred_sentiment.view(-1, 3)

        # Model = FNNMultitaskModelの場合は有効にする
        # persona = persona.repeat(1, data.shape[1], 1)


        loss_persona = loss_function1(pred_persona, persona)

        loss_sentiment = loss_function2(pred_sentiment, y_sentiment)

        if not train:
            for i in range(5):
                tmp_loss = round(loss_function1(pred_persona[:, i], persona[:, i]).item(), 3)
                separate_loss.append(tmp_loss)
        
        loss = loss_weight * loss_persona + (1-loss_weight) * loss_sentiment

        pred_sentiment = torch.argmax(pred_sentiment, dim=1)

        # 学習ログ
        persona_losses.append(loss_persona.item())
        sentiment_losses.append(loss_sentiment.item())
        all_losses.append(loss.item())
        pred_personas.append(pred_persona.data.cpu().numpy())
        pred_sentiments.append(pred_sentiment.data.cpu().numpy())
        y_personas.append(persona.data.cpu().numpy())
        y_sentiments.append(y_sentiment.data.cpu().numpy())


        if train:
            loss.backward()
            # if args.tensorboard:
            #     for param in model.named_parameters():
            #         writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step() 

    avg_all_loss = round(np.sum(all_losses)/len(all_losses), 4)
    avg_persona_loss = round(np.sum(persona_losses)/len(persona_losses), 4)
    avg_sentiment_loss = round(np.sum(sentiment_losses)/len(sentiment_losses), 4)

    return avg_all_loss, avg_persona_loss, avg_sentiment_loss, pred_personas, pred_sentiments, y_personas, y_sentiments, separate_loss


def objective(trial):

    D_i = 1218

    # ハイパラチューニング対象
    D_h = int(trial.suggest_discrete_uniform("D_h", 100, 1000, 50))
    D_o = int(trial.suggest_discrete_uniform("D_o", 10, 500, 10))
    in_droprate = trial.suggest_discrete_uniform("in_droprate", 0.0, 0.5, 0.05)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-10, 1e-3)
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
    loss_weight = trial.suggest_discrete_uniform('alpha', 0.05, 0.95, 0.05)

    config = dict(trial.params) 
    config['D_i'] = D_i
    config['trial.number'] = trial.number

    config.update(vars(args))

    testfiles = []
    for f in glob.glob('../data/Hazumi1911/dumpfiles/*.csv'):
        testfiles.append(os.path.splitext(os.path.basename(f))[0])

    testfiles = sorted(testfiles)

    all_loss = []
    sentiment_loss = []
    persona_loss = []
    accuracy = []

    extr_loss = [] 
    agre_loss = [] 
    cons_loss = [] 
    neur_loss = [] 
    open_loss = [] 

    pos = []
    neu = []
    neg = []

    project_name = randomname(10)


    for testfile in tqdm(testfiles, position=0, leave=True):

        model = LSTMMultitaskModel(D_i, D_h, D_o,n_classes=3, dropout=in_droprate)
        loss_function1 = nn.MSELoss() # 性格特性
        loss_function2 = nn.CrossEntropyLoss() # 心象

        wandb.init(project=project_name, config=config, name=testfile)

                    
        if args.cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)

        train_loader, valid_loader, test_loader = get_Hazumi_loaders(testfile, batch_size=args.batch_size, valid=0.1) 

        best_persona_loss, best_persona_label, best_persona_pred = None, None, None
        best_sentiment_loss, best_sentiment_label, best_sentiment_pred = None, None, None 
        best_all_loss = None

        best_val_loss = None

        es = EarlyStopping(patience=10, verbose=1)


        for epoch in range(args.epochs):
            trn_all_loss, trn_persona_loss, trn_sentiment_loss, _, _, _, _, _ = train_or_eval_model(model, loss_function1, loss_function2, train_loader, epoch, optimizer, True, loss_weight=loss_weight)
            val_all_loss, val_persona_loss, val_sentiment_loss, _, _, _, _, _ = train_or_eval_model(model, loss_function1, loss_function2, valid_loader, epoch, loss_weight=loss_weight)
            tst_all_loss, tst_persona_loss, tst_sentiment_loss, tst_persona_pred, tst_sentiment_pred, \
            tst_persona_label, tst_sentiment_label, tst_persona_sep_loss = train_or_eval_model(model, loss_function1, loss_function2, test_loader, epoch, loss_weight=loss_weight)


            if best_all_loss == None or best_val_loss > val_persona_loss:
                best_persona_loss, best_persona_label, best_persona_pred, best_persona_sep_loss = \
                tst_persona_loss, tst_persona_label, tst_persona_pred, tst_persona_sep_loss

                best_sentiment_loss, best_sentiment_label, best_sentiment_pred = \
                tst_sentiment_loss, tst_sentiment_label, tst_sentiment_pred

                best_all_loss = tst_all_loss
                best_val_loss = val_persona_loss
            
            if es(val_persona_loss):
                break

            wandb.log({
                "train loss": trn_all_loss,
                "train persona loss": trn_persona_loss, 
                "train sentiment loss": trn_sentiment_loss,
                "valid loss": val_all_loss,
                "valid persona loss": val_persona_loss, 
                "valid sentiment loss": val_sentiment_loss,
                "test loss": tst_all_loss,
                "test persona loss": tst_persona_loss, 
                "test sentiment loss": tst_sentiment_loss,
                "extraversion": tst_persona_sep_loss[0],
                "agreeableness": tst_persona_sep_loss[1],
                "conscientiousness": tst_persona_sep_loss[2],
                "neuroticism": tst_persona_sep_loss[3],
                "openness": tst_persona_sep_loss[4],
            })
            
        all_loss.append(best_all_loss)
        sentiment_loss.append(best_sentiment_loss)
        persona_loss.append(best_persona_loss)

        extr_loss.append(best_persona_sep_loss[0])
        agre_loss.append(best_persona_sep_loss[1])
        cons_loss.append(best_persona_sep_loss[2])
        neur_loss.append(best_persona_sep_loss[3]) 
        open_loss.append(best_persona_sep_loss[4])

        best_sentiment_pred = list(itertools.chain.from_iterable(best_sentiment_pred))
        best_sentiment_label = list(itertools.chain.from_iterable(best_sentiment_label))     

        acc = balanced_accuracy_score(best_sentiment_label, best_sentiment_pred)
        accuracy.append(acc)

        wandb.log({
            'best loss': best_all_loss,
            'best persona loss': best_persona_loss, 
            'best sentiment loss': best_sentiment_loss, 
            'best etra loss': best_persona_sep_loss[0],
            'best agre loss': best_persona_sep_loss[1],
            'best cons loss': best_persona_sep_loss[2], 
            'best neur loss': best_persona_sep_loss[3], 
            'best open loss': best_persona_sep_loss[4],
            'balanced accuracy': acc
        })

        matrix = confusion_matrix(best_sentiment_label, best_sentiment_pred)
        tmp = matrix.sum(axis=1)
        if len(matrix) == 3:
            neg.append(matrix[0][0] / tmp[0])
            neu.append(matrix[1][1] / tmp[1])
            pos.append(matrix[2][2] / tmp[2])
        else:
            neu.append(matrix[0][0] / tmp[0])
            pos.append(matrix[1][1] / tmp[1]) 


        # best_persona_pred = list(itertools.chain.from_iterable(best_persona_pred))
        # best_persona_label = list(itertools.chain.from_iterable(best_persona_label))

        wandb.finish()

    wandb.init(project=project_name, config=args, name='stats')
    wandb.log({
    '0best loss': np.array(all_loss).mean(),
    '0best persona loss': np.array(persona_loss).mean(),
    '0best sentiment loss': np.array(sentiment_loss).mean(),
    '1best etra loss': np.array(extr_loss).mean(),
    '2best agre loss': np.array(agre_loss).mean(),
    '3best cons loss': np.array(cons_loss).mean(), 
    '4best neur loss': np.array(neur_loss).mean(), 
    '5best open loss': np.array(open_loss).mean(),
    '6balanced accuracy': np.array(accuracy).mean()
    })
    wandb.finish()

    print('=====Result=====')
    print(f'低群：{np.nanmean(neg)}') 
    print(f'中群：{np.nanmean(neu)}') 
    print(f'高群：{np.nanmean(pos)}')

    print("外向性   :", np.array(extr_loss).mean())
    print("協調性   :", np.array(agre_loss).mean())
    print("勤勉性   :", np.array(cons_loss).mean())
    print("神経症傾向:", np.array(neur_loss).mean())
    print("開放性   :", np.array(open_loss).mean())

    print(f'損失（全体）： {np.array(all_loss).mean():.3f}')
    print(f'損失（心象）： {np.array(sentiment_loss).mean():.3f}')
    print(f'損失（性格特性）： {np.array(persona_loss).mean():.3f}')

    print(f'正解率： {np.array(accuracy).mean():.5f}')
    
    return np.array(persona_loss).mean()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=600, metavar='E', help='number of epochs')
 
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


    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))