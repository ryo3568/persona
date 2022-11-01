import numpy as np
import argparse, time, pickle
import os
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import LSTMSentimentModel
from dataloader import HazumiMultiTaskDataset
from utils.EarlyStopping import EarlyStopping


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset) 
    idx = list(range(size)) 
    split = int(valid*size) 
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_Hazumi_loaders(test_file, batch_size=32, valid=0.1, target=5, num_workers=0, pin_memory=False):
    trainset = HazumiMultiTaskDataset(test_file, target=target)
    testset = HazumiMultiTaskDataset(test_file, target=target, train=False, scaler=trainset.scaler) 

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
    preds = [] 
    personas = [] 
    sentiments = []
    assert not train or optimizer!=None 
    if train:
        model.train() 
    else:
        model.eval() 
    for data in dataloader:
        if train:
            optimizer.zero_grad() 
        
        text, visual, audio, persona, sentiment =\
        [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        personas.append(persona.data.cpu().numpy())

        persona = persona.repeat(1, text.shape[1], 1)
        sentiment = sentiment.view(text.shape[0], -1, 1)

        # data = audio
        # data = torch.cat((visual, audio), dim=-1)
        data = torch.cat((text, visual, audio), dim=-1)

        if not train:
            seq_len = int(rate*data.shape[1])
            data = data[:, :seq_len, :]
            

        pred = model(data)


        loss1 = loss_function1(pred[:, :, :5], persona)
        loss2 = loss_function2(pred[:, :, 5:], sentiment)

        loss = loss1 + loss2

        preds.append(pred.data.cpu().numpy())

        # personas.append(persona.data.cpu().numpy())
        sentiments.append(sentiment.data.cpu().numpy())
        persona_losses.append(loss1.item())
        sentiment_losses.append(loss2.item())

        if train:
            loss1.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step() 


        
    # if preds != []:
    #     preds = np.concatenate(preds)
    #     personas = np.concatenate(personas) 
    #     sentiments = np.concatenate(sentiments)
    # else:
    #     return float('nan'), [], []

    avg_persona_loss = round(np.sum(persona_losses)/len(persona_losses), 4)
    avg_sentiment_loss = round(np.sum(sentiment_losses)/len(sentiment_losses), 4)

    return avg_persona_loss, avg_sentiment_loss, personas, sentiments, preds



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
    parser.add_argument('--target', type=int, default=5, help='0:extraversion, 1:agreauleness, 2:conscientiousness, 3:neuroticism, 4:openness, 5:all')

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
    target = args.target
    
    # if target == 5:
    #     n_classes = 5
    # else:
    #     n_classes = 1
    
    n_classes = 6

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
            model = LSTMSentimentModel(D_i, D_h, D_o,n_classes=n_classes, dropout=args.dropout)
            

            if cuda:
                model.cuda()
            
            loss_function1 = nn.MSELoss() # 性格特性
            loss_function2 = nn.MSELoss() # 心象
            # loss_function = nn.MaskedMSELoss()

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

            train_loader, valid_loader, test_loader = get_Hazumi_loaders(testfile, batch_size=batch_size, valid=0.1, target=target) 

            best_persona_loss, best_persona_label, best_pred = None, None, None

            es = EarlyStopping(patience=10, verbose=1)

            for epoch in range(n_epochs):
                # start_time = time.time() 
                train_persona_loss, train_sentiment_loss, _, _, _= train_or_eval_model(model, loss_function1, loss_function2, train_loader, epoch, optimizer, True)
                valid_persona_loss, valid_sentiment_loss, _, _, _= train_or_eval_model(model, loss_function1, loss_function2, valid_loader, epoch)
                test_persona_loss, _, test_persona_label, _, test_pred = train_or_eval_model(model, loss_function1, loss_function2, test_loader, epoch, rate=rate)


                if best_persona_loss == None or best_persona_loss > test_persona_loss:
                    best_persona_loss, best_persona_label, best_pred = \
                    test_persona_loss, test_persona_label, test_pred

                if args.tensorboard:
                    writer.add_scalar('test: loss', test_persona_loss, epoch) 
                    writer.add_scalar('train: loss', train_persona_loss, epoch) 
                
                if es(valid_persona_loss):
                    break

            if args.tensorboard:
                writer.close() 

            loss.append(best_persona_loss)

        losses.append(np.array(loss).mean())

    print('Result')
    print('損失：', np.array(losses).mean())
    print(losses)

