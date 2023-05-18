import torch
import random 
import string 
import os
import glob
import collections
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd 
from sklearn.cluster import KMeans

def count_parameters(model):
    params = 0 
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel() 
    return params

def randomname(n):
   return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def get_files(version):
    testfiles1911 = []
    testfiles2010 = []
    testfiles2012 = []

    for f in glob.glob('../data/Hazumi/Hazumi1911/dumpfiles/*.csv'):
        testfiles1911.append(os.path.splitext(os.path.basename(f))[0])
    for f in glob.glob('../data/Hazumi/Hazumi2010/dumpfiles/*.csv'):
        testfiles2010.append(os.path.splitext(os.path.basename(f))[0])
    for f in glob.glob('../data/Hazumi/Hazumi2012/dumpfiles/*.csv'):
        testfiles2012.append(os.path.splitext(os.path.basename(f))[0])

    if version == "1911":
        testfiles = sorted(testfiles1911)
    elif version == "2010":
        testfiles = sorted(testfiles2010)
    elif version == "2012":
        testfiles = sorted(testfiles2012)
    elif version == "all":
        testfiles = []
        testfiles.extend(testfiles1911)
        testfiles.extend(testfiles2010)
        testfiles.extend(testfiles2012)
        testfiles = sorted(testfiles)

    return testfiles

def get_traits():
    return ['外向性', '協調性', '勤勉性', '神経症傾向', '開放性']

def calc_confusion(pred, label):
    Pred = [[], [], [], [], []]
    Label = [[], [], [], [], []]

    for i in range(len(pred)):
        for j in range(5):
            Pred[j].append(1 if pred[i][j] >= 0.5 else 0)
            Label[j].append(label[i][j])

    Traits = ['extr', 'agre', 'cons', 'neur', 'open']

    for i, trait in enumerate(Traits):
        print(trait)
        print(confusion_matrix(Label[i], Pred[i]))

def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def clustering(data, vid, n_clusters=4):
    TP_cluster = {}
    trait_name = ["外向性", "協調性", "勤勉性", "神経症傾向", "開放性"]
    df = pd.DataFrame.from_dict(data, orient='index', columns=trait_name)

    model = KMeans(n_clusters=n_clusters,  random_state=1)
    model.fit(df) 
    cluster = model.labels_
    TP_cluster = dict(zip(vid, cluster))

    return TP_cluster


def rolling_window(x, window_size, step_size):
    seq_len = x.shape[1]
    if window_size == -1:
        window_size = seq_len
    return torch.stack([x[:,i: i+window_size, :] for i in range(0, seq_len-window_size+1, step_size)])

def dict_standardize(data, vid):
    columns = ['E', 'A', 'C', 'N', 'O']
    df = pd.DataFrame.from_dict(data, orient='index', columns=columns)
    sc = StandardScaler()
    df = sc.fit_transform(df)
    df = pd.DataFrame(df, columns=columns, index=vid).round(2)

    res = {}
    for i, r in df.iterrows():
        res[i] = r.values.tolist() 
    return res

def Normalization(x):
    res = {}
    for id, X in x.items():
        norm_X = []
        for x in X:
            norm_X.append(round((x-2)/12, 2))
        res[id] = norm_X
    return res

def calc_acc(label, pred):
    count = [0, 0, 0]
    sum = [0, 0, 0]
    for l, p in zip(label, pred):
        if l == p:
            count[l] += 1 
            sum[l] += 1 
        else:
            sum[l] += 1
    return count[0], count[1], count[2], sum[0], sum[1], sum[2]
