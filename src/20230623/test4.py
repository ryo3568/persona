import pickle
import argparse
from tqdm import tqdm
import numpy as np 
import pandas as pd
from sklearn import svm 
from sklearn import preprocessing 
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import wandb 
import utils

# from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.simplefilter('ignore')


def load_data(testfile, n_clusters):
    train_data = []
    test_data = []
    path = f'../data/Hazumi_features/Hazumi1911_features.pkl'
    SS, TS, SP, TP, Text, _, _, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    users = []
    cluster = utils.clusteringv2(TP, n_clusters=n_clusters)

    # cluster = utils.clustering(SP, testfile, n_clusters=n_clusters)

    for file in vid:
        text = pd.DataFrame(Text[file])
        ss = pd.DataFrame(SS[file])
        data = pd.concat([text, ss], axis=1)

        if file == testfile:
            test_data.append(data)
        elif cluster[file] == cluster[testfile]:
            train_data.append(data)
            users.append(file)
    
    # print(users)

    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data) 
    train_data = train_data.sample(frac=1, random_state=1)

    x_train, y_train, x_test, y_test =\
         train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values, test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

    return x_train, y_train, x_test, y_test, cluster


if __name__ == "__main__":

    # plt.rcParams['font.family'] = "Noto Serif CJK JP"   
    # plt.rcParams['figure.figsize'] = [20, 3]

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="1911")
    parser.add_argument('--n_clusters', type=int, default=26)

    args = parser.parse_args()

    files = utils.get_files(args.version)

    config = {
        "C": 100,
        "gamma": 0.0001,
        "kernel": "sigmoid",
    }

    pred_all = []
    true_all = []

    results = [[] for _ in range(args.n_clusters)]

    for testfile in files:

        try:

            x_train, y_train, x_test, y_test, cluster = load_data(testfile, n_clusters=args.n_clusters)

            model = svm.SVC(C=config["C"], gamma=config["gamma"], kernel=config["kernel"]) 


            model.fit(x_train, y_train) 
            pred = model.predict(x_test)

            print(f"{testfile}({cluster[testfile]}) : {round(accuracy_score(y_test, pred), 3)}")
            results[cluster[testfile]].append(round(accuracy_score(y_test, pred), 3))

            pred_all = np.concatenate([pred_all, pred])
            true_all = np.concatenate([true_all, y_test])
        except:
            print(testfile,  ": Error")

    print("============= Results =============")
    for i, result in enumerate(results):
        try:
            print(f"{round(sum(result) / len(result), 3)}({len(result)})")
        except:
            print("Error")
    # print(classification_report(true_all, pred_all))
    print("accuracy: ", round(accuracy_score(true_all, pred_all), 3))