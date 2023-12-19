import argparse
import pickle 
import yaml
import datetime
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.simplefilter('ignore', FutureWarning)

from utils import profiling, fix_seed

def load_data(testuser, modal, pmode):
    path = f'../data/Hazumi_features/Hazumi1911_features.pkl'
    _, SS_binary, SS_ternary, _, TS_binary, TS_ternary, SP, TP, Text, Audio, Visual, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    columns = ['E', 'A', 'C', 'N', 'O']
    df = pd.DataFrame.from_dict(TP, orient='index', columns=columns)
    _df = (df - df.mean() ) / df.std(ddof=0)

    X_train = [] 
    Y_train = []

    if args.ss:
        if args.binary:
            label = SS_binary
        else:
            label = SS_ternary
    else:
        if args.binary:
            label = TS_binary
        else:
            label = TS_ternary

    
    test_profile = profiling(pmode, testuser,  _df)

    for user in vid:
        user_profile = profiling(pmode, user, _df)
        data = [] 
        if 't' in modal:
            text = pd.DataFrame(Text[user])
            data.append(text)
        if 'a' in modal:
            audio = Audio[user]
            stds = StandardScaler()
            audio = stds.fit_transform(audio)
            audio = pd.DataFrame(audio)
            data.append(audio)
        if 'v' in modal:
            visual = Visual[user]
            stds = StandardScaler()
            visual = stds.fit_transform(visual)
            visual = pd.DataFrame(visual)
            data.append(visual)
        data = pd.concat(data, axis=1).values
        if user == testuser:
            X_test = data
            Y_test = label[user]
        elif user_profile == test_profile:
            X_train.extend(data)
            Y_train.extend(label[user])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    '''
    0. 前準備
    '''
    fix_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--modal', type=str, default="t")
    parser.add_argument('--pmode', type=int, default=0)
    parser.add_argument('--ss', action='store_true', default=False)
    parser.add_argument('--binary', action='store_true', default=False)
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--balanced', action="store_true", default=False, help="class_weight is balanced")
    args = parser.parse_args()

    config = {}
    config["annot"] = "SS" if args.ss else "TS"
    config["label"] = "binary" if args.binary else "ternary"
    config["modal"] = args.modal
    config["pmode"] = args.pmode

    print(config)

    path = '../data/Hazumi_features/Hazumi1911_features.pkl'
    _, _, _, _, _, _, _, _, _, _, _, ids = pickle.load(open(path, 'rb'), encoding='utf-8')

    results = {}

    for test_user in tqdm(ids):
        '''
        1. データの準備
        '''
        x_train, y_train, x_test, y_test = load_data(test_user, args.modal, args.pmode)

        '''
        2. モデルの構築
        '''
        if args.balanced:
            model = svm.SVC(C=100, gamma=0.0001, kernel="sigmoid", class_weight="balanced")
        else:
            model = svm.SVC(C=100, gamma=0.0001, kernel="sigmoid")
        
        # model = MLPClassifier()
        '''
        3. モデルの学習
        '''
        model.fit(x_train, y_train)

        '''
        4. モデルの評価
        '''
        preds = model.predict(x_test)

        acc = accuracy_score(y_test, preds) 
        # f1 = f1_score(y_test, preds, average="weighted")

        results[test_user] = float(round(acc, 3))

    config = {}
    config["annot"] = "SS" if args.ss else "TS"
    config["label"] = "binary" if args.binary else "ternary"
    config["modal"] = args.modal
    config["pmode"] = args.pmode
    # config["task"] = "regresion" if args.regression else "classification"
    
    yml = {}
    yml["config"] = config 
    yml["results"] = results

    timestamp = datetime.datetime.now().strftime('%m%d%H%M%S')

    if args.save_results:
        file_name = f'results/{timestamp}.yaml'
        with open(file_name, 'w') as f:
            yaml.dump(yml, f)