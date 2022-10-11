'''
class-weightの実装
loss_functionをCrossEntropyLoss -> MaskedNLLLossに変更
実験を複数回回せるように変更(default:5回)
tqdmで学習の進捗状況をわかりやすくした
args.rateの削除
'''

import numpy as np
import argparse, time, pickle
import os
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import LSTMSentimentModel2, MaskedNLLLoss
from dataloader import Hazumi1911SentimentDataset
from utils.EarlyStopping import EarlyStopping


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset) 
    idx = list(range(size)) 
    split = int(valid*size) 
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_Hazumi1911_loaders(test_file, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = Hazumi1911SentimentDataset(test_file)
    testset = Hazumi1911SentimentDataset(test_file, train=False, scaler=trainset.scaler) 

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
    masks = []
    assert not train or optimizer!=None 
    if train:
        model.train() 
    else:
        model.eval() 
    for data in dataloader:
        if train:
            optimizer.zero_grad() 
        
        text, visual, audio, persona, mask, label =\
        [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        persona = persona.repeat(1, text.shape[1], 1)

        # data = audio
        # data = torch.cat((visual, audio), dim=-1)
        data = torch.cat((text, visual, audio), dim=-1)

        log_prob = model(data)

        lp_ = log_prob.view(-1, log_prob.size()[2])
        label_ = label.view(-1)
        
        loss = loss_function(lp_, label_, mask)

        pred_ = torch.argmax(lp_, 1)

        preds.append(pred_.data.cpu().numpy())
        labels.append(label_.data.cpu().numpy())
        masks.append(mask.view(-1).cpu().numpy())
        losses.append(loss.item())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step() 
        
    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels) 
        masks = np.concatenate(masks)
    else:
        return float('nan'), [], [], []

    # avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_loss = round(np.sum(losses)/len(losses), 4)

    return avg_loss, labels, preds, masks



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
    
    n_classes = 3

    D_i = 3063
    D_h = 100 
    D_o = 100

    testfiles = []
    for f in glob.glob('../data/Hazumi1911/dumpfiles/*.csv'):
        testfiles.append(os.path.splitext(os.path.basename(f))[0])

    testfiles = sorted(testfiles)

    negatives = [] # 低群の精度
    accuracies = [] # 感情認識精度
    best_losses = None 

    for i in range(args.iter):

        print(f'Iteration {i+1} / {args.iter}')
    
        accuracy = []
        negative = [] 

        for testfile in tqdm(testfiles, position=0, leave=True):
            model = LSTMSentimentModel2(D_i, D_h, D_o,n_classes=n_classes, dropout=args.dropout)

            if cuda:
                model.cuda()
    
            # loss_function = nn.CrossEntropyLoss()
            loss_weights = torch.FloatTensor([13.70, 2.77, 1.77])

            if args.class_weight:
                loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
            else:
                loss_function = MaskedNLLLoss()

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

            train_loader, valid_loader, test_loader = get_Hazumi1911_loaders(testfile, batch_size=batch_size, valid=0.1) 

            best_loss, best_label, best_pred, best_mask = None, None, None, None 

            test_losses = []

            es = EarlyStopping(patience=10, verbose=1)

            for epoch in range(n_epochs):
                # start_time = time.time() 
                train_loss, _, _, _ = train_or_eval_model(model, loss_function, train_loader, epoch, optimizer, True)
                valid_loss, _, _, _ = train_or_eval_model(model, loss_function, valid_loader, epoch)
                test_loss, test_label, test_pred, test_mask = train_or_eval_model(model, loss_function, test_loader, epoch)

                test_losses.append(valid_loss)

                if best_loss == None or best_loss > test_loss:
                    best_loss, best_label, best_pred, best_mask = \
                    test_loss, test_label, test_pred, test_mask

                if args.tensorboard:
                    writer.add_scalar('test: loss', test_loss, epoch) 
                    writer.add_scalar('train: loss', train_loss, epoch) 
                
                if es(valid_loss):
                    break

            if args.tensorboard:
                writer.close() 

            if best_losses == None or best_losses > best_loss:
                best_losses = best_loss 
                best_model = model
                best_model_iter = i + 1

            # print('Test performance..')
            # print('Testfile name {}'.format(testfile))
            # print('Loss {} F1-score {}'.format(best_loss, round(f1_score(best_label, best_pred, sample_weight=best_mask, average='weighted')*100, 2)))
            # print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
            # print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
            # score = accuracy_score(best_label, best_pred, sample_weight=best_mask)
            # print('accuracy(weight) : ', accuracy_score(best_label, best_pred, sample_weight=best_mask))
            
            acc_score = accuracy_score(best_label, best_pred, sample_weight=best_mask)
            accuracy.append(acc_score)

            cm = confusion_matrix(best_label, best_pred, sample_weight=best_mask)
            if cm.ndim == 3:
                negative += cm[0][0] # 低群の予測に成功したデータ数

            # class_report = classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, output_dict=True, zero_division=0)
            # if '0' in class_report:
            #     negative.append(class_report['0']['precision'])
            
        torch.save(best_model.state_dict(), '../data/Hazumi1911/model/model.pt')
        negatives.append(negative)
        accuracies.append(np.array(accuracy).mean())

    print('Result')
    print('低群の精度：', np.array(negatives).mean() / 178)
    print(negatives)
    print('感情認識精度：', np.array(accuracies).mean())
    print(accuracies)
    print(f"Best Model is Iter{best_model_iter}'s model")


