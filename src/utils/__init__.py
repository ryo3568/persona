import random 
import numpy as np
import torch

def fix_seed(seed=123):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def profiling(profile, id, TP=None, age_th=4):
    age = int(id[5])
    gender = id[4]

    per_th = 0

    if profile == 0:
        res = 0
    elif profile == 1:
        if gender == 'F':
            res = 0
        else:
            res = 1
    elif profile == 2:
        if age <= age_th:
            res = 0 
        else:
            res = 1
    elif profile == 3:
        if gender == 'F':
            if age <= age_th:
                res = 0
            else:
                res = 1
        else:
            if age <= age_th:
                res = 2 
            else:
                res = 3
    elif profile == 4:
        res = 1 if TP.loc[id, :][0] >= per_th else 0
    elif profile == 5:
        res = 1 if TP.loc[id, :][1] >= per_th else 0
    elif profile == 6:
        res = 1 if TP.loc[id, :][2] >= per_th else 0
    elif profile == 7:
        res = 1 if TP.loc[id, :][3] >= per_th else 0
    elif profile == 8:
        res = 1 if TP.loc[id, :][4] >= per_th else 0
    elif profile == 9:
        if age <= 3:
            res = 0 
        elif age <= 5:
            res = 1
        else:
            res = 2
    elif profile == 10:
        res = age - 2
    return res
