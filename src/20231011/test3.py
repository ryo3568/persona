import os
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
import datetime

# from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.simplefilter('ignore')

import sys
sys.path.append("../")
import utils

def clustering(name):
    age = int(name[5])
    return age - 2


def load_data(testfile):

    test_profile = clustering(testfile)

    train_data = []
    test_data = []
    path = f'../../data/Hazumi_features/Hazumi1911_features.pkl'
    SS, _, _, _, Text, Audio, Visual, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    for file in vid:

        # user_profile = int(file[5])
        user_profile = clustering(file)

        data = []

        text = pd.DataFrame(Text[file])
        data.append(text)

        audio = pd.DataFrame(Audio[file])
        data.append(audio)

        visual = pd.DataFrame(Visual[file])
        data.append(visual)

        ss = pd.DataFrame(SS[file])
        data.append(ss)

        # data = pd.concat([text, audio, visual, ss], axis=1)
        data = pd.concat(data, axis=1)

        if file == testfile:
            test_data.append(data)
        elif test_profile == user_profile:
            train_data.append(data)
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data) 
    train_data = train_data.sample(frac=1, random_state=1)


    x_train, y_train, x_test, y_test =\
         train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values, test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
        
    mm = preprocessing.MinMaxScaler() 
    # mm = preprocessing.StandardScaler()

    x_train = mm.fit_transform(x_train)
    x_test = mm.transform(x_test)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="1911")

    args = parser.parse_args()

    files = utils.get_files(args.version)

    config = {
        "C": 100,
        "gamma": 0.0001,
        "kernel": "sigmoid",
    }
    
    dt_now = datetime.datetime.now()
    dt_now_str = dt_now.strftime("%m%d%H%M%S")
    filename = os.path.basename(__file__)
    filename = os.path.splitext(filename)[0]
    resultsfile = f"./results/{filename}-{dt_now_str}.txt"

    pred_all = []
    true_all = []

    pred_cluster = [[], [], [], [], [], []]
    true_cluster = [[], [], [], [], [], []]

    for testfile in files:
        try:
            # test_age = int(testfile[5])
            # test_profile = test_age - 2
            test_profile = clustering(testfile)

            x_train, y_train, x_test, y_test = load_data(testfile)

            # model = svm.SVC(C=config["C"], gamma=config["gamma"], kernel=config["kernel"]) 
            model = svm.SVC(C=config["C"], gamma=config["gamma"], kernel=config["kernel"], class_weight='balanced') 
            model.fit(x_train, y_train) 
            pred = model.predict(x_test)

            print(f"{testfile}")
            print(f"{round(accuracy_score(y_test, pred), 3)}")
            print(confusion_matrix(y_test, pred))
            # print(f"{round(accuracy_score(y_test, pred), 3)}")

            pred_all = np.concatenate([pred_all, pred])
            true_all = np.concatenate([true_all, y_test])

            pred_cluster[test_profile] = np.concatenate([pred_cluster[test_profile], pred])
            true_cluster[test_profile] = np.concatenate([true_cluster[test_profile], y_test])


        except Exception as e:
            print(f"{testfile}:Error")
            print('type:' + str(type(e)))
            print('args:' + str(e.args))
            # print('message:' + e.message)
            print('e自身:' + str(e))
        print('=' * 15)

    print("============= Results(ALL) =============")
    print("accuracy: ", round(accuracy_score(true_all, pred_all), 3))
    print("f1      : ", round(f1_score(true_all, pred_all), 3))
    print("confusion: ", confusion_matrix(true_all, pred_all))

    for i in range(len(true_cluster)):
        print(f"============== Results ( {i} ) ================")
        print("accuracy: ", round(accuracy_score(true_cluster[i], pred_cluster[i]), 3))
        print("f1      : ", round(f1_score(true_cluster[i], pred_cluster[i]), 3))

    # with open(resultsfile, mode='a') as f:
    #     f.write(f"{testfile}: {round(best_f1, 3)}\n")
    #     for ans, conf in zip(best_ans, best_conf):
    #         for i, id in enumerate(ans):
    #             if i:
    #                 f.write(",")
    #             f.write(f"{id}")
    #         f.write("\n")
    #         for i, c in enumerate(conf):
    #             for j, d in enumerate(c):
    #                 if j:
    #                     f.write(", ")
    #                 f.write(f"{str(d)}")
    #             f.write("\n")
    #         f.write("------------\n")
    #     f.write("=====================================\n")
