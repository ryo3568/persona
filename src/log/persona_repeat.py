import numpy as np
import argparse, time, pickle
import os
import glob
import csv
from tqdm import tqdm
import itertools 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import LSTMModel, FNNModel
from dataloader import HazumiDataset
from utils.EarlyStopping import EarlyStopping


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


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False, rate=1.0):
    losses = []
    preds = [] 
    labels = []
    test_loss = [] 
    time_pred = []
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
            for i in range(text.size()[1]):
                pred = model(data[:, :i+1, :])
                # Model = Netの場合は有効にする
                # _persona = persona.repeat(1, i+1, 1) 
                # loss = loss_function(pred, _persona)
                loss = loss_function(pred, persona)
                test_loss.append(round(loss.item(),3)) 
                time_pred.append(pred.squeeze().tolist())

            
        
        pred = model(data)
        # Model = Netの場合は有効にする
        # persona = persona.repeat(1, data.shape[1], 1) 

        loss = loss_function(pred, persona)
        

        # 学習ログ
        losses.append(loss.item())
        preds.append(pred.data.cpu().numpy())
        labels.append(persona.data.cpu().numpy())


        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step() 


    avg_loss = round(np.sum(losses)/len(losses), 4)

    return avg_loss, preds, labels, test_loss, time_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.15, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weight')
    parser.add_argument('--attention', action='store_true', default=False, help='use attention on top of lstm')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    # 追加
    parser.add_argument('--pretrained', action='store_true', default=False, help='use pretrained parameter')
    parser.add_argument('--rate', type=float, default=1.0, help='number of sequence length')
    parser.add_argument('--iter', type=int, default=5, help='number of experiments')
    
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

    D_i = 1218
    D_h = 256
    D_o = 32

    testfiles = []
    for f in glob.glob('../data/Hazumi1911/dumpfiles/*.csv'):
        testfiles.append(os.path.splitext(os.path.basename(f))[0])

    testfiles = sorted(testfiles)

    losses = []
    time_losses = []

    for i in range(args.iter):

        print(f'Iteration {i+1} / {args.iter}')

        loss = []
        time_loss = []
        ave_time_loss = [] 
        std_time_loss = [] 

        for testfile in tqdm(testfiles, position=0, leave=True):

            model = LSTMModel(D_i, D_h, D_o,n_classes=5, dropout=args.dropout)
            loss_function = nn.MSELoss()

            if args.pretrained:
                model.load_state_dict(torch.load(f'../data/model/{testfile}.pt'), strict=False)

                        
            if cuda:
                model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

            train_loader, valid_loader, test_loader = get_Hazumi_loaders(testfile, batch_size=batch_size, valid=0.1) 

            best_loss, best_label, best_pred, best_time_loss= None, None, None, None

            best_val_loss = None

            # es = EarlyStopping(patience=10, verbose=1)

            for epoch in range(n_epochs):
                trn_loss, _, _, _, _ = train_or_eval_model(model, loss_function, train_loader, epoch, optimizer, True)
                val_loss, _, _, _, _ = train_or_eval_model(model, loss_function, valid_loader, epoch)
                tst_loss, tst_pred, tst_label, tst_time_loss, tst_time_pred= train_or_eval_model(model, loss_function, test_loader, epoch, rate=rate)
            

                if best_loss == None or best_val_loss > val_loss:
                    best_loss, best_label, best_pred, best_time_loss, best_time_pred = \
                    tst_loss, tst_label, tst_pred, tst_time_loss, tst_time_pred 

                    best_val_loss = val_loss

                if args.tensorboard:
                    writer.add_scalar('test: loss', tst_loss, epoch) 
                    writer.add_scalar('train: loss', trn_loss, epoch) 

                
                # if es(val_loss):
                #     break


            dir = f'../data/results/{testfile}/'
            if not os.path.exists(dir):
                os.makedirs(dir)

            with open(dir+'single221206.csv', 'w') as f:
                writer = csv.writer(f) 
                writer.writerow(best_time_loss) 
                writer.writerow(best_time_pred)
                writer.writerow(best_label[0])

            if args.tensorboard:
                writer.close() 

            loss.append(best_loss)
            time_loss.append(best_time_loss)
            ave_time_loss.append(np.mean(best_time_loss)) 
            std_time_loss.append(np.std(best_time_loss))
    


            # best_pred = list(itertools.chain.from_iterable(best_pred))
        print(f'損失：{loss}')
        print("======リアルタイム推定結果======")
        print(f'平均：{np.mean(ave_time_loss)}')
        print(ave_time_loss)
        print(f'標準偏差：{np.mean(std_time_loss)}')
        print(std_time_loss)


        losses.append(np.array(loss).mean())


    print('=====Result=====')
    print(f'損失： {np.array(losses).mean():.3f}')
    print(losses)