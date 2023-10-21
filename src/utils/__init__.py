import random 
import os
import glob
import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch


def get_files(version='1911'):
    testfiles1712 = []
    testfiles1902 = []
    testfiles1911 = []
    testfiles2010 = []
    testfiles2012 = []

    for f in glob.glob('../data/Hazumi/Hazumi1712/dumpfiles/*.csv'):
        testfiles1712.append(os.path.splitext(os.path.basename(f))[0])
    for f in glob.glob('../data/Hazumi/Hazumi1902/dumpfiles/*.csv'):
        testfiles1902.append(os.path.splitext(os.path.basename(f))[0])
    for f in glob.glob('../../data/Hazumi/Hazumi1911/dumpfiles/*.csv'):
        testfiles1911.append(os.path.splitext(os.path.basename(f))[0])
    for f in glob.glob('../data/Hazumi/Hazumi2010/dumpfiles/*.csv'):
        testfiles2010.append(os.path.splitext(os.path.basename(f))[0])
    for f in glob.glob('../data/Hazumi/Hazumi2012/dumpfiles/*.csv'):
        testfiles2012.append(os.path.splitext(os.path.basename(f))[0])

    if version == "1712":
        testfiles = sorted(testfiles1712)
    elif version == "1902":
        testfiles = sorted(testfiles1902)
    elif version == "1911":
        testfiles = sorted(testfiles1911)
    elif version == "2010":
        testfiles = sorted(testfiles2010)
    elif version == "2012":
        testfiles = sorted(testfiles2012)
    elif version == "all":
        testfiles = []
        testfiles.extend(testfiles1712)
        testfiles.extend(testfiles1902)
        testfiles.extend(testfiles1911)
        testfiles.extend(testfiles2010)
        testfiles.extend(testfiles2012)
        testfiles = sorted(testfiles)
    return testfiles


def profiling(id, mode):
    age = int(id[5])
    gender = id[4]

    if mode == 0:
        res = 0
    elif mode == 1:
        # 性別
        if gender == 'F':
            res = 0
        else:
            res = 1
    elif mode == 2:
        # 年齢 40 <=, 40 >
        if age <= 4:
            res = 0
        else:
            res = 1
    elif mode == 3:
        # 年齢 30 <=, 50 <=, 50 > 
        if age <= 3:
            res = 0 
        elif age <= 5:
            res = 1
        else:
            res = 2
    elif mode == 4: 
        # 年齢 20, 30, 40, 50, 60, 70
        res = age - 2
    return res

def clusteringv1(data, testfile, n_clusters=4):
    columns = ['E', 'A', 'C', 'N', 'O']
    test_data = {}
    test_data[testfile] = data.pop(testfile)
    # df = pd.DataFrame.from_dict(data, orient='index', columns=columns)
    df = pd.DataFrame.from_dict(data, orient='index')
    # test_df = pd.DataFrame.from_dict(test_data, orient='index', columns=columns)
    test_df = pd.DataFrame.from_dict(test_data, orient='index')
    # sc = StandardScaler()
    index = df.index
    # df = sc.fit_transform(df)
    # test_df = sc.transform(test_df)
    # df = pd.DataFrame(df, columns=columns)
    # df = pd.DataFrame(df)
    # df.index = index
    # test_df = pd.DataFrame(test_df, columns=columns)
    # test_df = pd.DataFrame(test_df)
    # test_df.index = [testfile]

    model = KMeans(n_clusters=n_clusters, random_state=0) 
    model.fit(df)
    pred = model.predict(df)

    cluster = {} 
    for i, id in enumerate(index):
        cluster[id] = pred[i]
    cluster[testfile] = model.predict(test_df)[0]

    dist = model.transform(test_df)
    dist = round(dist[0][cluster[testfile]], 3)

    return cluster, dist

def clusteringv2(data, n_clusters=4):
    df = pd.DataFrame.from_dict(data, orient='index')
    index = df.index

    model = KMeans(n_clusters=n_clusters, random_state=0) 
    model.fit(df)
    pred = model.predict(df)

    cluster = {} 
    for i, id in enumerate(index):
        cluster[id] = pred[i]

    return cluster

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

def torch_fix_seed(seed=123):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
