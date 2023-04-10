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

def get_files():
    testfiles = []
    for f in glob.glob('../data/Hazumi1911/dumpfiles/*.csv'):
        testfiles.append(os.path.splitext(os.path.basename(f))[0])

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


