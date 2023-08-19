import pickle
import argparse
from tqdm import tqdm
import numpy as np 
import pandas as pd
from sklearn import svm 
from sklearn import preprocessing 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import wandb 
import copy

# from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.simplefilter('ignore')

import sys
sys.path.append("../")
import utils


def load_data(testfile, data_files):
    train_data = []
    test_data = []
    path = f'../../data/Hazumi_features/Hazumi1911_features.pkl'
    SS, _, _, TP, Text, _, _, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    for file in vid:
        text = pd.DataFrame(Text[file])
        ss = pd.DataFrame(SS[file])
        data = pd.concat([text, ss], axis=1)

        if file == testfile:
            test_data.append(data)
        elif file in data_files:
            train_data.append(data)
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data) 
    train_data = train_data.sample(frac=1, random_state=1)

    x_train, y_train, x_test, y_test =\
         train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values, test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    # plt.rcParams['font.family'] = "Noto Serif CJK JP"   
    # plt.rcParams['figure.figsize'] = [20, 3]

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="1911")

    args = parser.parse_args()

    # files = utils.get_files(args.version)
    files = ["1911F2001", "1911F3001", "1911F4001", "1911F5001", "1911F6001", "1911F7002",
                "1911M2001", "1911M4001", "1911M5001", "1911M6001", "1911M7001"]

    config = {
        "C": 100,
        "gamma": 0.0001,
        "kernel": "sigmoid",
    }

    results = []
    for testfile in files:
        filenames = copy.deepcopy(files)
        filenames.remove(testfile)
        try:
            x_train, y_train, x_test, y_test = load_data(testfile, filenames)

            model = svm.SVC(C=config["C"], gamma=config["gamma"], kernel=config["kernel"]) 

            model.fit(x_train, y_train) 
            pred = model.predict(x_test)

            # acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred)
            conf = confusion_matrix(y_test, pred)
            print(f"{testfile}: {f1}")
            print(conf)
            results.append(f1)
        except:
            pass
    
    print("======Results======")
    print(round(sum(results) / len(results), 3))