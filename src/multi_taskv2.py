import numpy as np
import argparse, time, pickle
import os
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import LSTMMultiTaskModel, LSTMMultiTaskModelv2
from dataloader import HazumiMultiTaskDataset
from utils.EarlyStopping import EarlyStopping


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset) 
    idx = list(range(size)) 
    split = int(valid*size) 
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_Hazumi_loaders(test_file, batch_size=32, valid=0.1, args=None, num_workers=2, pin_memory=False):
    trainset = HazumiMultiTaskDataset(test_file, args=args)
    testset = HazumiMultiTaskDataset(test_file, train=False, scaler=trainset.scaler, args=args) 

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


def train_or_eval_model(model, loss_function1, loss_function2, dataloader, epoch, optimizer=None, train=False, rate=1.0):
    persona_losses = []
    sentiment_losses = []
    all_losses = [] 
    pred_personas = []
    pred_sentiments = [] 
    y_personas = []
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
        [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        # data = audio
        # data = torch.cat((visual, audio), dim=-1)
        data = torch.cat((text, visual, audio), dim=-1)

        if not train:
            seq_len = int(rate*data.shape[1])
            data = data[:, :seq_len, :]
            
        
        pred_persona, pred_sentiment = model(data)

        if not args.regression:
            s_ternary = s_ternary.view(-1)
            y_sentiment = s_ternary
            pred_sentiment = pred_sentiment.view(-1, 3)
        else:
            sentiment = sentiment.view(text.shape[0], -1, 1)
            y_sentiment = sentiment


        y_persona = torch.repeat_interleave(persona, repeats=text.shape[1], dim=0).view(-1, text.shape[1], 5)
        loss_persona = loss_function1(pred_persona, y_persona)

        loss_sentiment = loss_function2(pred_sentiment, y_sentiment)
        
        loss = loss_persona + loss_sentiment

        if not args.regression:
            pred_sentiment = torch.argmax(pred_sentiment, dim=1)

        # 学習ログ
        persona_losses.append(loss_persona.item())
        sentiment_losses.append(loss_sentiment.item())
        all_losses.append(loss.item)
        pred_personas.append(pred_persona.data.cpu().numpy())
        pred_sentiments.append(pred_sentiment.data.cpu().numpy())
        y_personas.append(y_persona.data.cpu().numpy())
        y_sentiments.append(y_sentiment.data.cpu().numpy())

        # print('-----------------------')
        # print(pred_persona.size())
        # print(pred_sentiment.size())
        # print(y_persona.size())
        # print(y_sentiment.size())

        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step() 


    # if pred_personas != []:
    #     pred_personas = np.concatenate(pred_pe)
    #     personas = np.concatenate(personas) 
    #     sentiments = np.concatenate(sentiments)
        
    # else:
    #     return float('nan'), [], []

    avg_persona_loss = round(np.sum(persona_losses)/len(persona_losses), 4)
    avg_sentiment_loss = round(np.sum(sentiment_losses)/len(sentiment_losses), 4)

    return all_losses, avg_persona_loss, avg_sentiment_loss, pred_personas, pred_sentiments, y_personas, y_sentiments

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
    parser.add_argument('--rate', type=float, default=1.0, help='number of sequence length')
    parser.add_argument('--iter', type=int, default=5, help='number of experiments')
    parser.add_argument('--regression', action='store_true', default=False, help='estimating sentiment with regression model')
    parser.add_argument('--persona_first_annot', action='store_true', default=False, help='using persona label annotated by user')
    parser.add_argument('--sentiment_first_annot', action='store_true', default=False, help='using sentiment label annotated by user')
    
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda 
    if args.cuda:
        print('Running on GPU') 
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter 
        writer = SummaryWriter()

    batch_size = args.batch_size 
    cuda = args.cuda 
    n_epochs = args.epochs 
    rate = args.rate
    
    n_classes = 5

    D_i = 3063
    D_h = 100 
    D_o = 100

    testfiles = []
    for f in glob.glob('../data/Hazumi1911/dumpfiles/*.csv'):
        testfiles.append(os.path.splitext(os.path.basename(f))[0])

    testfiles = sorted(testfiles)

    losses = []

    for i in range(args.iter):

        print(f'Iteration {i+1} / {args.iter}')

        loss = []

        for testfile in tqdm(testfiles, position=0, leave=True):

            if not args.regression:

                model = LSTMMultiTaskModelv2(D_i, D_h, D_o,n_classes=n_classes, dropout=args.dropout)
                loss_function2 = nn.CrossEntropyLoss() # 心象
            else:
                model = LSTMMultiTaskModel(D_i, D_h, D_o,n_classes=n_classes, dropout=args.dropout) 
                loss_function2 = nn.MSELoss() # 心象

            loss_function1 = nn.MSELoss() # 性格特性

                        
            if cuda:
                model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

            train_loader, valid_loader, test_loader = get_Hazumi_loaders(testfile, batch_size=batch_size, valid=0.1, args=args) 

            best_persona_loss, best_persona_label, best_pred = None, None, None

            # es = EarlyStopping(patience=10, verbose=1)

            for epoch in range(n_epochs):
                trn_persona_loss, trn_sentiment_loss, _, _, _, _, _= train_or_eval_model(model, loss_function1, loss_function2, train_loader, epoch, optimizer, True)
                val_persona_loss, val_sentiment_loss, _, _, _, _, _= train_or_eval_model(model, loss_function1, loss_function2, valid_loader, epoch)
                tst_all_loss, tst_persona_loss, tst_sentiment_loss, tst_persona_pred, tst_sentiment_pred, \
                tst_persona_label, tst_sentiment_label = train_or_eval_model(model, loss_function1, loss_function2, test_loader, epoch, rate=rate)


                if best_persona_loss == None or best_persona_loss > tst_persona_loss:
                    best_persona_loss, best_persona_label, best_pred = \
                    tst_persona_loss, tst_persona_label, tst_persona_pred
                    print(epoch)

                if args.tensorboard:
                    writer.add_scalar('test: loss', tst_persona_loss, epoch) 
                    writer.add_scalar('train: loss', trn_persona_loss, epoch) 
                
                # if es(val_persona_loss):
                #     break


            if args.tensorboard:
                writer.close() 

            loss.append(best_persona_loss)

        losses.append(np.array(loss).mean())

    print('=====Result=====')
    print(f'損失： {np.array(losses).mean():.3f}')
    print(losses)