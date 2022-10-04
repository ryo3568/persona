import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import os 
import glob

def make_label(filename):
    df = pd.read_excel('../data/Hazumi1911/questionnaire/1911questionnaires.xlsx', sheet_name=4, index_col=0, header=1)
    data = df.loc[filename, :].values.tolist()
    return [data[0]+(8-data[5]), (8-data[1])+data[6], data[2]+(8-data[7]), data[3]+(8-data[8]), data[4]+(8-data[9])]

def make_label_thirdbigfive(filename):
    df = pd.read_excel('../data/Hazumi1911/questionnaire/220818thirdbigfive-Hazumi1911.xlsx', sheet_name=5, header=1, index_col=0)
    data = df.loc[filename].values.tolist()
    return [data[5], data[13], data[21], data[29], data[37]]

videoIDs = {}
videoAudio = {}
videoText = {}
videoVisual = {} 
videoLabels_persona = {}
videoLabels_sentiment = {}
videoSentence = {}

Vid = []

path = '../data/Hazumi1911/dumpfiles/*'

files = glob.glob(path)

for file_path in sorted(files):
    filename = os.path.basename(file_path).split('.', 1)[0]
    df = pd.read_csv(file_path)
    text = df.loc[:, 'word#0001':'su'].values.tolist()
    audio = df.loc[:, 'pcm_RMSenergy_sma_max':'F0_sma_de_kurtosis'].values.tolist()
    visual = df.loc[:, '17_acceleration_max':'AU45_c_mean'].values.tolist()
    label = df.loc[:, 'TS_ternary'].values.tolist()
    
    Vid.append(filename)
    videoAudio[filename] = audio 
    videoText[filename] = text 
    videoVisual[filename] = visual 
    videoIDs[filename] = []
    videoLabels_persona[filename] = make_label_thirdbigfive(filename)
    videoLabels_sentiment[filename] = label
    videoSentence[filename] = []

with open('../data/Hazumi1911/Hazumi1911_features/Hazumi1911_features_persona.pkl', mode='wb') as f:
    pickle.dump((videoIDs, videoLabels_persona, videoText, videoAudio, videoVisual, videoSentence, Vid), f)

with open('../data/Hazumi1911/Hazumi1911_features/Hazumi1911_features_sentiment.pkl', mode='wb') as f:
    pickle.dump((videoIDs, videoLabels_sentiment, videoText, videoAudio, videoVisual, videoSentence, Vid), f)