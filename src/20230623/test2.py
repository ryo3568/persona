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


def load_data(testfile):
    train_data = []
    test_data = []
    path = f'../data/Hazumi_features/Hazumi1911_features.pkl'
    _, TS, _, _, Text, _, _, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    for file in vid:
        text = pd.DataFrame(Text[file])
        ts = pd.DataFrame(TS[file])
        data = pd.concat([text, ts], axis=1)

        if file == testfile:
            test_data.append(data)
        else:
            train_data.append(data)

    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data) 
    train_data = train_data.sample(frac=1, random_state=0)

    x_train, y_train, x_test, y_test =\
         train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values, test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

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

    pred_all = []
    true_all = []

    for testfile in files:

        try:

            x_train, y_train, x_test, y_test = load_data(testfile)

            model = svm.SVC(C=config["C"], gamma=config["gamma"], kernel=config["kernel"]) 


            model.fit(x_train, y_train) 
            pred = model.predict(x_test)

            print(testfile, ": ", round(accuracy_score(y_test, pred), 3))

            pred_all = np.concatenate([pred_all, pred])
            true_all = np.concatenate([true_all, y_test])
        except:
            print(testfile,  ": Error")

    
    print("============= Results =============")
    # print(classification_report(true_all, pred_all))
    print("accuracy: ", round(accuracy_score(true_all, pred_all), 3))