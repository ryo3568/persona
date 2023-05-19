import pickle
import argparse
from tqdm import tqdm
import numpy as np 
import pandas as pd
from sklearn import svm 
from sklearn import preprocessing 
import matplotlib.pyplot as plt
import wandb 
import utils

import warnings
warnings.simplefilter('ignore')


def load_data(testfile):
    train_data = []
    test_data = []
    path = '../data/Hazumi_features/Hazumi1911_features.pkl'
    SS, _, _, _, text, audio, visual, vid \
        = pickle.load(open(path, 'rb'), encoding='utf-8')
    for file in vid:
        data_t = pd.DataFrame(text[file])
        data_a = pd.DataFrame(audio[file])
        data_v = pd.DataFrame(visual[file])
        label = pd.DataFrame(SS[file])

        data = pd.concat([data_t, data_a, data_v, label], axis=1)
        if file == testfile:
            test_data.append(data)
        else:
            train_data.append(data)
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data) 
    train_data = train_data.sample(frac=1, random_state=0)

    x_train, y_train, x_test, y_test =\
         train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values, test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

    # scaler = preprocessing.MinMaxScaler()
    scaler = preprocessing.StandardScaler()

    x_train = scaler.fit_transform(x_train) 
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test

# def draw_heatmap(data, filename,row_labels, column_labels):
#     fig, ax = plt.subplots() 
#     heatmap = ax.pcolor(data, cmap=plt.cm.Reds)

#     ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)  
#     ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)


#     ax.invert_yaxis() 
#     ax.xaxis.tick_top() 

#     ax.set_xticklabels(column_labels, minor=False) 
#     ax.set_yticklabels(row_labels, minor=False)

#     ax.set_xlabel("発話番号")
#     ax.xaxis.set_label_position('top')

#     plt.title(filename)
#     plt.show() 
    # plt.savefig('image.png')


if __name__ == "__main__":

    # plt.rcParams['font.family'] = "Noto Serif CJK JP"   
    # plt.rcParams['figure.figsize'] = [20, 3]

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--version', type=str, default="1911")
    args = parser.parse_args()

    files = utils.get_files(args.version)

    config = {
        "C": 100,
        "gamma": 0.0001,
        "kernel": "sigmoid"
    }

    project_name = 'svm' 
    group_name = utils.randomname(5)

    for testfile in files:
        if args.wandb:
            wandb.init(project=project_name, group=group_name, config=config, name=testfile)

        x_train, y_train, x_test, y_test = load_data(testfile)

        model = svm.SVC(C=config["C"], gamma=config["gamma"], kernel=config["kernel"]) 
        model.fit(x_train, y_train) 

        print(model.score(x_train, y_train))
        print(model.score(x_test, y_test))

        # p_proba = model.predict_proba(x_test)
        # ans = y_test[0]
        # result.append(p_proba[:, ans])

        # result.append(1 if model.score(x_test, y_test) >= 0.5 else 0)

        # result = np.stack(result)
        # seq_len = result.shape[1]
        # draw_heatmap(result, testfile, traits, list(range(1, seq_len+1)))

        if args.wandb:
            wandb.log({
                "acc": model.score(x_test, y_test)
            })

            wandb.finish()