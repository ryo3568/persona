import pickle
import argparse
from tqdm import tqdm
import numpy as np 
import pandas as pd
from sklearn import svm 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.simplefilter('ignore')

import sys
sys.path.append("../")
import utils


def load_data(testfile):
    path = '../../data/Hazumi_features/Hazumi1911_features.pkl'
    SS, _, _, _, Text, _, _, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    ss = SS[testfile]
    text = Text[testfile]

    x_train, x_test, y_train, y_test = train_test_split(text, ss, test_size=0.2)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    # plt.rcParams['font.family'] = "Noto Serif CJK JP"   
    # plt.rcParams['figure.figsize'] = [20, 3]

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="1911")

    args = parser.parse_args()

    files = utils.get_files(args.version)

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

    for testfile in files:

        try:
            x_train, y_train, x_test, y_test = load_data(testfile)

            # model = svm.SVC(C=config["C"], gamma=config["gamma"], kernel=config["kernel"], class_weight="balanced") 
            model = svm.SVC(C=config["C"], gamma=config["gamma"], kernel=config["kernel"]) 
            model.fit(x_train, y_train) 
            pred = model.predict(x_test)

            # c = [0, 0, 0, 0]
            # for y in y_train:
            #     if y == 0:
            #         c[0] += 1 
            #     else:
            #         c[1] += 1

            # for y in y_test:
            #     if y == 0:
            #         c[2] += 1
            #     else:
            #         c[3] += 1

            # print(f"{testfile} : {round(accuracy_score(y_test, pred), 3)} [({c[0]},{c[1]}), ({c[2]}, {c[3]})])")
            print(f"{testfile} : {round(accuracy_score(y_test, pred), 3)}")
            print(confusion_matrix(y_test, pred))

            pred_all = np.concatenate([pred_all, pred])
            true_all = np.concatenate([true_all, y_test])
        except:
            print(testfile, ": Error")

    print("============= Results =============")
    print("accuracy: ", round(accuracy_score(true_all, pred_all), 3))