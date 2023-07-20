import pickle
import argparse
from tqdm import tqdm
import numpy as np 
import pandas as pd
from sklearn import svm 
from sklearn import preprocessing 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import wandb 

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

    files = utils.get_files(args.version)

    config = {
        "C": 100,
        "gamma": 0.0001,
        "kernel": "sigmoid",
    }

    for testfile in files:
        filenames = files
        filenames.remove(testfile)
        n = len(filenames)
        results = []
        best_acc = 0.0
        for i in range(1, 2 ** n):
            try:
                data_files = []
                for j in range(n):
                    if ((i >> j) & 1):
                        data_files.append(filenames[j])
                x_train, y_train, x_test, y_test = load_data(testfile, data_files)

                model = svm.SVC(C=config["C"], gamma=config["gamma"], kernel=config["kernel"]) 

                model.fit(x_train, y_train) 
                pred = model.predict(x_test)

                acc = accuracy_score(y_test, pred)
                conf = confusion_matrix(y_test, pred)
                if acc >= best_acc:
                    if acc > best_acc:
                        best_acc = acc
                        best_ans = [data_files]
                        best_conf = [conf]
                    else:
                        best_ans.append(data_files)
                        best_conf.append(confusion_matrix(y_test, pred))
            except:
                pass
        # with open("./results/test5.txt", mode='a') as f:
        #     f.write(f"{testfile}: {round(best_acc), 3}")
        #     for ans, conf in zip(best_ans, best_conf):
        #         f.write(ans)
        #         f.write(conf)
        #     f.write("=====================================")
        with open("./results/test5.txt", mode='a') as f:
            f.write(f"{testfile}: {round(best_acc, 3)}\n")
            for ans, conf in zip(best_ans, best_conf):
                for i, id in enumerate(ans):
                    if i:
                        f.write(",")
                    f.write(f"{id}")
                f.write("\n")
                for i, c in enumerate(conf):
                    for j, d in enumerate(c):
                        if j:
                            f.write(", ")
                        f.write(f"{str(d)}")
                    f.write("\n")
                f.write("------------\n")
            f.write("=====================================\n")
