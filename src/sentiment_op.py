import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import optuna
from model import LSTMSentimentModel
from dataloader import HazumiDataset
from utils.EarlyStopping import EarlyStopping

import warnings 
warnings.simplefilter('ignore')


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
    sentiment_losses = []
    pred_sentiments = [] 
    y_sentiments = []
    assert not train or optimizer!=None 
    if train:
        model.train() 
    else:
        model.eval() 
    for data in dataloader:
        if train:
            optimizer.zero_grad() 
        
        text, visual, audio, persona, sentiment, s_ternary =\
        [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]

        # data = audio
        # data = torch.cat((visual, audio), dim=-1)
        data = torch.cat((text, visual, audio), dim=-1)
            
        
        pred_sentiment = model(data)


        s_ternary = s_ternary.view(-1)
        y_sentiment = s_ternary
        pred_sentiment = pred_sentiment.view(-1, 3)
        # Model = FNNMultitaskModelの場合は有効にする
        # persona = persona.repeat(1, data.shape[1], 1)

        loss_sentiment = loss_function(pred_sentiment, y_sentiment)

        pred_sentiment = torch.argmax(pred_sentiment, dim=1)

        # 学習ログ
        sentiment_losses.append(loss_sentiment.item())
        pred_sentiments.append(pred_sentiment.data.cpu().numpy())
        y_sentiments.append(y_sentiment.data.cpu().numpy())


        if train:
            loss_sentiment.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step() 

    avg_sentiment_loss = round(np.sum(sentiment_losses)/len(sentiment_losses), 4)

    return avg_sentiment_loss, pred_sentiments, y_sentiments

def objective(trial):
    D_i = 1218 

    # ハイパラチューニング対象
    D_h = int(trial.suggest_discrete_uniform("D_h", 200, 500, 50))
    D_o = int(trial.suggest_discrete_uniform("D_o", 10, 32, 2))
    in_droprate = trial.suggest_discrete_uniform("in_droprate", 0.0, 0.2, 0.05)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-10, 1e-3)
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)

    testfiles = []
    for f in glob.glob('../data/Hazumi1911/dumpfiles/*.csv'):
        testfiles.append(os.path.splitext(os.path.basename(f))[0])

    testfiles = sorted(testfiles)

    loss = []
    accuracy = []

    pos = []
    neu = []
    neg = []


    for testfile in tqdm(testfiles, position=0, leave=True):

        model = LSTMSentimentModel(D_i, D_h, D_o,n_classes=3, dropout=in_droprate)
        loss_function = nn.CrossEntropyLoss() # 心象

                    
        if args.cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)

        train_loader, valid_loader, test_loader = get_Hazumi_loaders(testfile, batch_size=args.batch_size, valid=0.1) 

        best_loss, best_label, best_pred, best_model = None, None, None, None

        best_val_loss = None

        # es = EarlyStopping(patience=10, verbose=1)

        if args.tensorboard:
            from tensorboardX import SummaryWriter 
            writer = SummaryWriter()

        for epoch in range(args.epochs):
            trn_loss, _, _= train_or_eval_model(model, loss_function, train_loader, epoch, optimizer, True)
            val_loss, _, _= train_or_eval_model(model, loss_function, valid_loader, epoch)
            tst_loss, tst_pred, tst_label = train_or_eval_model(model, loss_function, test_loader, epoch)


            if best_loss == None or best_val_loss > val_loss:
                best_loss, best_label, best_pred, best_model = \
                tst_loss, tst_label, tst_pred, model.state_dict()

                best_val_loss = val_loss

            if args.tensorboard:
                writer.add_scalar('test: loss', tst_loss, epoch) 
                writer.add_scalar('train: loss', trn_loss, epoch) 

            if args.save_model:
                torch.save(best_model, f'../data/model/{testfile}.pt')
        
        # if es(val_persona_loss):
        #     break


        if args.tensorboard:
            writer.close() 

        loss.append(best_loss)


        best_pred = list(itertools.chain.from_iterable(best_pred))
        best_label = list(itertools.chain.from_iterable(best_label))          


        accuracy.append(balanced_accuracy_score(best_label, best_pred))

        matrix = confusion_matrix(best_label, best_pred)
        tmp = matrix.sum(axis=1)
        if len(matrix) == 3:
            neg.append(matrix[0][0] / tmp[0])
            neu.append(matrix[1][1] / tmp[1])
            pos.append(matrix[2][2] / tmp[2])
        else:
            neu.append(matrix[0][0] / tmp[0])
            pos.append(matrix[1][1] / tmp[1])                  
            

    print('=====Result=====')
    print(f'低群：{np.nanmean(neg)}') 
    print(f'中群：{np.nanmean(neu)}') 
    print(f'高群：{np.nanmean(pos)}')


    print(f'損失（心象）： {np.array(loss).mean():.3f}')

    print(f'正解率： {np.array(accuracy).mean():.5f}')
    
    return np.array(loss).mean()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.25, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weight')
    parser.add_argument('--attention', action='store_true', default=False, help='use attention on top of lstm')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    # 追加
    parser.add_argument('--trail_size', type=int, default=1, help='number of trail')
     
    args = parser.parse_args()

    # print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda 
    if args.cuda:
        print('Running on GPU') 
    else:
        print('Running on CPU')

    study = optuna.create_study() 
    study.optimize(objective, n_trials=args.trail_size)

    print(study.best_params)