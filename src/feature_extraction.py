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


def calc_thirdpersona(filename, normalized=False):
    """
    アノテータの質問回答から性格特性スコアを算出
    """
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

def making_TP_binary(TP, vid, standard=False):
    """
    性格特性スコアを2クラスに分類
    """
    TP_binary = {}
    df = pd.DataFrame.from_dict(TP, orient='index')

    if standard:
        sc = StandardScaler()
        df = sc.fit_transform(df)
        df = pd.DataFrame(df, index=vid) 
        df = (df >= 0 )* 1
    else:
        df = (df >= 8) * 1

    for id in vid:
        TP_binary[id] = df.loc[id, :].tolist()

    return TP_binary

def feature_extract():

    text = {}
    bert_text = {}
    audio = {}
    visual = {} 

    TP = {}
    TP_norm = {}
    TP_binary = None 
    TP_binary_stand = None
    TP_cluster = None

    TS_ternary = {}
    TS = {}

    vid = []

    for file_path in tqdm(sorted(files)):
        filename = os.path.basename(file_path).split('.', 1)[0]
        df = pd.read_csv(file_path)
        start = df['start(exchange)[ms]'].values.tolist()

        vid.append(filename)
        bert_text[filename] = extract_sentence(filename, start)
        text[filename] = df.loc[:, 'word#0001':'su'].values.tolist()
        audio[filename] = df.loc[:, 'pcm_RMSenergy_sma_max':'F0_sma_de_kurtosis'].values.tolist()
        visual[filename] = df.loc[:, '17_acceleration_max':'AU45_c_mean'].values.tolist()

        TP[filename] = calc_thirdpersona(filename)
        TP_norm[filename] = calc_thirdpersona(filename, True)
        TS_ternary[filename] = df.loc[:, 'TS_ternary'].astype(int).values.tolist()
        TS[filename] = df.loc[:, 'TS1':'TS5'].mean(axis='columns').values.tolist()

    TP_binary = making_TP_binary(TP, vid)
    TP_binary_stand = making_TP_binary(TP, vid, standard=True)

    TP_cluster = utils.clustering(vid, TP, n_clusters=4)

    with open(feature_path + 'Hazumi1911_features.pkl', mode='wb') as f:
        pickle.dump((TS, TS_ternary, TP, TP_binary_stand, TP_cluster, bert_text, audio, visual, vid), f)

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

    feature_extract()
