import pickle
import argparse
from tqdm import tqdm
import numpy as np 
import pandas as pd
from sklearn import svm 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')

import sys
sys.path.append("../")
import utils


def load_data(file, testfile):
    path = '../../data/Hazumi_features/Hazumi1911_features.pkl'
    SS, _, _, _, Text, _, _, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    x_train = Text[file]
    y_train = SS[file]

    x_test = Text[testfile]
    y_test = SS[testfile]

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    # plt.rcParams['font.family'] = "Noto Serif CJK JP"   
    # plt.rcParams['figure.figsize'] = [20, 3]

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="1911")
    parser.add_argument('--test', type=str, default="1911F2001")

    args = parser.parse_args()

    files = utils.get_files(args.version)
    testfile = args.test

    config = {
        "C": 100,
        "gamma": 0.0001,
        "kernel": "sigmoid",
    }

    # if config["self_s"]:
    #     project_name = 'selfsentiment-svm' 
    # else:
    #     project_name= 'thirdsentiment-svm'

    pred_all = []
    true_all = []

    for file in files:
        if file == testfile:
            print(f"{file[4:]}:-")
            continue
        try:
            x_train, y_train, x_test, y_test = load_data(file, testfile)

            model = svm.SVC(C=config["C"], gamma=config["gamma"], kernel=config["kernel"]) 
            model.fit(x_train, y_train) 
            pred = model.predict(x_test)

            print(f"{file[4:]}:{round(accuracy_score(y_test, pred), 3)}")
            print(confusion_matrix(y_test, pred))
            # print(f"{round(accuracy_score(y_test, pred), 3)}")

            pred_all = np.concatenate([pred_all, pred])
            true_all = np.concatenate([true_all, y_test])
        except:
            print(f"{file[4:]}:Error")

    print("============= Results =============")
    print("accuracy: ", round(accuracy_score(true_all, pred_all), 3))