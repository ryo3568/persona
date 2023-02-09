import re
import pickle 
import os 
import glob
from tqdm import tqdm
import pympi.Elan 
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch 
from transformers import BertModel 
from transformers import BertJapaneseTokenizer

import utils

from transformers import logging
logging.set_verbosity_error()

# 本人にアノテーションされた性格特性スコアの算出
def calc_persona(filename, normalized=False):
    df = pd.read_excel('../data/Hazumi1911/questionnaire/1911questionnaires.xlsx', sheet_name=4, index_col=0, header=1)
    data = df.loc[filename, :].values.tolist()
    res = [data[0]+(8-data[5]), (8-data[1])+data[6], data[2]+(8-data[7]), data[3]+(8-data[8]), data[4]+(8-data[9])]
    if normalized:
        return [(i-2)/12 for i in res]
    else:
        return res

# 第三者にアノテーションされた性格特性スコアの算出
def calc_thirdpersona(filename, normalized=False):
    df = pd.read_excel('../data/Hazumi1911/questionnaire/220818thirdbigfive-Hazumi1911.xlsx', sheet_name=5, header=1, index_col=0)
    data = df.loc[filename].values.tolist()
    res = [data[5], data[13], data[21], data[29], data[37]]
    if normalized:
        return [(i-2)/12 for i in res]
    else:
        return res

def preprocess(text):
    """ 
    フィラー記号(F *)，|の除去
    """
    text = re.sub('\([^)]*\)', '', text)
    text = text.replace('|', '')

    return text

def embedding(sentences):
    # 前処理

    sentences = [preprocess(text) for text in sentences]

    # BERTトークン化
    encoded = tokenizer.batch_encode_plus(
        sentences, padding=True, add_special_tokens=True
    )


    # BERTトークンID列を抽出
    input_ids = torch.tensor(encoded["input_ids"], device=device) 

    # BERTの最大許容トークン数が512なので超える場合は切り詰める
    input_ids = input_ids[:, :512] 

    with torch.no_grad():
        outputs = model(input_ids) 

    # 最終層の隠れ状態ベクトルを取得
    last_hidden_states = outputs[0] 

    # [CLS]トークンの単語ベクトルを抽出
    vecs = last_hidden_states[:, 0, :]

    return vecs.tolist()

def eaf_to_df( eaf: pympi.Elan.Eaf ) -> pd.DataFrame:
    tier_names = list( eaf.tiers.keys() )

    def timeslotid_to_time( timeslotid: str ) -> float:
        return eaf.timeslots[ timeslotid ] / 1000

    def parse( tier_name: str, tier: dict ) -> pd.DataFrame:
        values = [ (key,) + value[:-1] for key, value in tier.items() ]
        df = pd.DataFrame( values, columns=[ "id", "start", "end", "transcription"] )

        df["start"] = df["start"].apply( timeslotid_to_time )
        df["end"] = df["end"].apply( timeslotid_to_time )
        df["ID"] = df.apply( lambda x: f"{tier_name}-{x.name}", axis=1 )
        df = df.reindex( columns=["ID", "start", "end", "transcription"] )

        return df

    dfs = [ parse(tier_name=name, tier=eaf.tiers[name][0]) for name in tier_names ]
    df = pd.concat( dfs )
    df = df.sort_values( "start" )
    df = df.reset_index( drop=True )
    return df

def extract_sentence(filename, start):
    res = []
    src = '../data/Hazumi1911/elan/' + filename + '.eaf' 

    eaf = pympi.Elan.Eaf(src) 
    df = eaf_to_df(eaf) 

    df['start'] = (df['start'] * 1000).astype(int)

    for time in start:
        sentence = df[(df['start'] == time) & (df['ID'].str.contains('user'))]['transcription'].values.tolist()
        if len(sentence) == 0:
            sentence = ['']
        sentence = embedding(sentence)
        res.append(sentence[0])

    return res

def feature_extract():
    audio = {}
    text = {}
    visual = {} 

    third_persona = {}
    # persona = {}
    TS_ternary = {}
    # SS_ternary = {}
    third_sentiment = {}
    # sentiment = {}

    vid = []

    for file_path in tqdm(sorted(files)):
        filename = os.path.basename(file_path).split('.', 1)[0]
        df = pd.read_csv(file_path)

        vid.append(filename)
        text[filename] = df.loc[:, 'word#0001':'su'].values.tolist()
        audio[filename] = df.loc[:, 'pcm_RMSenergy_sma_max':'F0_sma_de_kurtosis'].values.tolist()
        visual[filename] = df.loc[:, '17_acceleration_max':'AU45_c_mean'].values.tolist()

        # persona[filename] = calc_persona(filename)
        third_persona[filename] = calc_thirdpersona(filename)
        TS_ternary[filename] = df.loc[:, 'TS_ternary'].values.tolist()
        # SS_ternary[filename] = df.loc[:, 'SS_ternary'].values.tolist()
        third_sentiment[filename] = df.loc[:, 'TS1':'TS5'].mean(axis='columns').values.tolist()
        # sentiment[filename] = df.loc[:, 'SS'].values.tolist()

    # ファイル書き込み
    with open(feature_path + 'Hazumi1911_features.pkl', mode='wb') as f:
        pickle.dump((TS_ternary, third_sentiment, third_persona, text, audio, visual, vid), f)

def feature_extract_bert():
    # 言語特徴量をBERTから抽出
    audio = {}
    text = {}
    visual = {} 

    third_persona = {}
    # persona = {}
    TS_ternary = {}
    # SS_ternary = {}
    third_sentiment = {}
    # sentiment = {}

    vid = []

    for file_path in tqdm(sorted(files)):
        filename = os.path.basename(file_path).split('.', 1)[0]
        df = pd.read_csv(file_path)
        start = df['start(exchange)[ms]'].values.tolist()

        vid.append(filename)
        text[filename] = extract_sentence(filename, start)
        audio[filename] = df.loc[:, 'pcm_RMSenergy_sma_max':'F0_sma_de_kurtosis'].values.tolist()
        visual[filename] = df.loc[:, '17_acceleration_max':'AU45_c_mean'].values.tolist()

        # persona[filename] = calc_persona(filename)
        third_persona[filename] = calc_thirdpersona(filename)
        TS_ternary[filename] = df.loc[:, 'TS_ternary'].values.tolist()
        # SS_ternary[filename] = df.loc[:, 'SS_ternary'].values.tolist()
        third_sentiment[filename] = df.loc[:, 'TS1':'TS5'].mean(axis='columns').values.tolist()
        # sentiment[filename] = df.loc[:, 'SS'].values.tolist()

    # ファイル書き込み
    with open(feature_path + 'Hazumi1911_features_bert.pkl', mode='wb') as f:
        pickle.dump((TS_ternary, third_sentiment, third_persona, text, audio, visual, vid), f)

def feature_extract_bert_norm():
    # 言語特徴量をBERTから抽出 && 性格特性スコア正規化
    audio = {}
    text = {}
    visual = {} 

    third_persona = {}
    # persona = {}
    TS_ternary = {}
    # SS_ternary = {}
    third_sentiment = {}
    # sentiment = {}

    vid = []

    for file_path in tqdm(sorted(files)):
        filename = os.path.basename(file_path).split('.', 1)[0]
        df = pd.read_csv(file_path)
        start = df['start(exchange)[ms]'].values.tolist()

        vid.append(filename)
        text[filename] = extract_sentence(filename, start)
        audio[filename] = df.loc[:, 'pcm_RMSenergy_sma_max':'F0_sma_de_kurtosis'].values.tolist()
        visual[filename] = df.loc[:, '17_acceleration_max':'AU45_c_mean'].values.tolist()

        # persona[filename] = calc_persona(filename, True)
        third_persona[filename] = calc_thirdpersona(filename, True)
        TS_ternary[filename] = df.loc[:, 'TS_ternary'].values.tolist()
        # SS_ternary[filename] = df.loc[:, 'SS_ternary'].values.tolist()
        third_sentiment[filename] = df.loc[:, 'TS1':'TS5'].mean(axis='columns').values.tolist()
        # sentiment[filename] = df.loc[:, 'SS'].values.tolist()

    # ファイル書き込み
    with open(feature_path + 'Hazumi1911_features_bert_norm.pkl', mode='wb') as f:
        pickle.dump((TS_ternary, third_sentiment, third_persona, text, audio, visual, vid), f)


def feature_extract_bert_stand(standard=False):
    # 言語特徴量をBERTから抽出
    audio = {}
    text = {}
    visual = {} 

    third_persona = {}
    # persona = {}
    TS_ternary = {}
    # SS_ternary = {}
    third_sentiment = {}
    # sentiment = {}

    vid = []

    for file_path in tqdm(sorted(files)):
        filename = os.path.basename(file_path).split('.', 1)[0]
        df = pd.read_csv(file_path)
        start = df['start(exchange)[ms]'].values.tolist()

        vid.append(filename)
        text[filename] = extract_sentence(filename, start)
        audio[filename] = df.loc[:, 'pcm_RMSenergy_sma_max':'F0_sma_de_kurtosis'].values.tolist()
        visual[filename] = df.loc[:, '17_acceleration_max':'AU45_c_mean'].values.tolist()

        # persona[filename] = calc_persona(filename)
        third_persona[filename] = calc_thirdpersona(filename)
        TS_ternary[filename] = df.loc[:, 'TS_ternary'].values.tolist()
        # SS_ternary[filename] = df.loc[:, 'SS_ternary'].values.tolist()
        third_sentiment[filename] = df.loc[:, 'TS1':'TS5'].mean(axis='columns').values.tolist()
        # sentiment[filename] = df.loc[:, 'SS'].values.tolist()

    plabel = {}
    df = pd.DataFrame.from_dict(third_persona, orient="index")

    if standard:
        sc = StandardScaler()
        df = sc.fit_transform(df)
        df = pd.DataFrame(df, index=vid) 
        df = (df >= 0 )* 1
    else:
        df = (df >= 8) * 1

    for id in vid:
        plabel[id] = df.loc[id, :].tolist()
    
    for k, v in TS_ternary.items():
        TS_ternary[k] = [int(x) for x in v]

    with open(feature_path + 'Hazumi1911_features_bert_standard.pkl', mode='wb') as f:
        pickle.dump((TS_ternary, third_sentiment, third_persona, plabel, text, audio, visual, vid), f)


def feature_extract_bert_cluster(n_cluster=4, standard=False):
    # 言語特徴量をBERTから抽出
    audio = {}
    text = {}
    visual = {} 

    third_persona = {}
    # persona = {}
    TS_ternary = {}
    # SS_ternary = {}
    third_sentiment = {}
    # sentiment = {}

    vid = []

    for file_path in tqdm(sorted(files)):
        filename = os.path.basename(file_path).split('.', 1)[0]
        df = pd.read_csv(file_path)
        start = df['start(exchange)[ms]'].values.tolist()

        vid.append(filename)
        text[filename] = extract_sentence(filename, start)
        audio[filename] = df.loc[:, 'pcm_RMSenergy_sma_max':'F0_sma_de_kurtosis'].values.tolist()
        visual[filename] = df.loc[:, '17_acceleration_max':'AU45_c_mean'].values.tolist()

        # persona[filename] = calc_persona(filename)
        third_persona[filename] = calc_thirdpersona(filename)
        TS_ternary[filename] = df.loc[:, 'TS_ternary'].values.tolist()
        # SS_ternary[filename] = df.loc[:, 'SS_ternary'].values.tolist()
        third_sentiment[filename] = df.loc[:, 'TS1':'TS5'].mean(axis='columns').values.tolist()
        # sentiment[filename] = df.loc[:, 'SS'].values.tolist()

    cluster = utils.clustering(third_persona, n_cluster)
    pcluster = dict(zip(vid, cluster))

    plabel = {}
    df = pd.DataFrame.from_dict(third_persona, orient="index")

    if standard:
        sc = StandardScaler()
        df = sc.fit_transform(df)
        df = pd.DataFrame(df, index=vid) 
        df = (df >= 0 )* 1
    else:
        df = (df >= 8) * 1

    for id in vid:
        plabel[id] = df.loc[id, :].tolist()

    for k, v in TS_ternary.items():
        TS_ternary[k] = [int(x) for x in v]

    # ファイル書き込み
    with open(feature_path + 'Hazumi1911_features_bert_cluster' + str(n_cluster) + '.pkl', mode='wb') as f:
        pickle.dump((TS_ternary, third_sentiment, third_persona, plabel, pcluster, text, audio, visual, vid), f)


if __name__ == '__main__':
    dumpfile_path = '../data/Hazumi1911/dumpfiles/*'
    files = glob.glob(dumpfile_path)
    feature_path = '../data/Hazumi_features/'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    # オークナイザーの読み込み
    tokenizer = BertJapaneseTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )

    # 学習済みモデルの読み込み
    model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking").to(device)

    # feature_extract()
    # feature_extract_bert()　# 言語特徴量としてBERTのエンコーディングを利用
    # feature_extract_bert_norm() # 性格特性スコアを[0,1]に正規化
    # feature_extract_bert_stand() # 性格特性スコアを標準化し、0以上と未満の2クラスに分類
    feature_extract_bert_cluster(n_cluster=2)