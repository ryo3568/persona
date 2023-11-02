import argparse
import pickle 
import yaml
import datetime
import math
from tqdm import tqdm
import numpy as np 
from sklearn import svm
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix

import sys 
sys.path.append("../")
from utils import profiling, fix_seed
from utils.EarlyStopping import EarlyStopping

def load_data(train, profile_vid):
    path = f'../../data/Hazumi_features/Hazumiall_features_binary.pkl'
    _, TS, _, _, Text, _, _, _ = pickle.load(open(path, 'rb'), encoding='utf-8')

    X_train = [] 
    Y_train = []

    for i, user in enumerate(profile_vid):
        label = TS[user]
        data = Text[user]
        if i in train:
            X_train.extend(data)
            Y_train.extend(label)
        else:
            X_test = data 
            Y_test = label
    
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
    parser.add_argument('--profile', type=int, default=0) # 0: ベースライン, 1: 性別, 2: 年齢(2クラス), 3: 年齢(3クラス), 4: 年齢(6クラス) 5: 性別 & 年齢(2クラス), 6: 性別 & 年齢(3クラス), 7: 性別 & 年齢(6クラス)
    parser.add_argument('--balanced', action="store_true", default=False, help="class_weight is balanced")
    args = parser.parse_args()

    path = f'../../data/Hazumi_features/Hazumiall_features_binary.pkl'
    _, _, _, _, _, _, _, vid = pickle.load(open(path, 'rb'), encoding='utf-8')

    if args.profile == 0:
        profile_num = 1 
    elif args.profile == 1:
        profile_num = 2 
    elif args.profile == 2:
        profile_num = 2
    elif args.profile == 3:
        profile_num = 3
    elif args.profile == 4:
        profile_num = 6
    elif args.profile == 5:
        profile_num = 4 
    elif args.profile == 6:
        profile_num = 6
    else:
        profile_num = 12

    Results = []

    kf = KFold(n_splits=3, shuffle=True, random_state=1)
    
    for profile_n in range(profile_num):
        print(f"profile number : {profile_n}")
        profile_vid = []
        for id in vid:
            user_profile = profiling(id, args.profile)
            if user_profile == profile_n:
                profile_vid.append(id)
        print(f"profile group member : {len(profile_vid)}")
        results = []

        for train, test in kf.split(profile_vid):
            '''
            1. データの準備
            '''
            x_train, y_train, x_test, y_test = load_data(train, profile_vid)

            '''
            2. モデルの構築
            '''
            if args.balanced:
                model = svm.SVC(C=100, gamma=0.0001, kernel="sigmoid", class_weight="balanced")
            else:
                model = svm.SVC(C=100, gamma=0.0001, kernel="sigmoid")

            '''
            3. モデルの学習
            '''
            model.fit(x_train, y_train)

            '''
            4. モデルの評価
            '''
            preds = model.predict(x_test)

            fpr, tpr, thresholds = roc_curve(y_test, preds)
            auc_score = auc(fpr, tpr)

            # acc = accuracy_score(y_test, preds) 
            # f1 = f1_score(y_test, preds, average="weighted")
            # f1 = f1_score(y_test, preds)

            results.append(auc_score)
            print("auc score : ", auc_score)
            print("confusion matrix : \n", confusion_matrix(y_test, preds))

        Results.append(round(sum(results) / len(results), 3))
        print("-" * 30)
        
        
    print('========== Results ==========')
    print(Results)
    Results_clean = []
    for result in Results:
        if not math.isnan(result):
            Results_clean.append(result)
    print(round(sum(Results_clean) / len(Results_clean), 3))

    # now = datetime.datetime.now()
    # if args.balanced:
    #     output_filename = f'results/binary/balanced/{args.profile}-' + now.strftime('%Y%m%d_%H%M%S') + '.yaml'
    # else:
    #     output_filename = f'results/binary/svm/{args.profile}-' + now.strftime('%Y%m%d_%H%M%S') + '.yaml'

    # with open(output_filename, 'w') as file:
    #     yaml.dump(results, file)