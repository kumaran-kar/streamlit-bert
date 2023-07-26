import pandas as pd
import TRTExtraction as trtext
#import PatentInfo as ptinfo
from ast import literal_eval
#import spacy


def TRTdf(abstracts,pubno,nlp):
    # abstracts = ptinfo.get_abstract(patent_path)
    # pubno = ptinfo.get_pubno(patent_path)
    ##df = pd.DataFrame()
    ##df['Publication number'] = pubno
    ##df['Abstarct'] = abstracts
    trts = list()
    # nlp = spacy.load(r"en_core_web_md-3.5.0")
    for abstract in abstracts:
        trt = trtext.get_trt(abstract,nlp)
        trts.append(trt)
    ##df['TRT'] = trts
    df_trt = pd.DataFrame()
    pubno1 = list()
    t1 = list()
    prep = list()
    t2 = list()
    iter = 0
    for row in trts :
        for trt in row:
            pubno1.append(pubno[iter])
            t1.append(trt['T1'])
            prep.append(trt['Preposition'])
            t2.append(trt['T2'])
        iter = iter + 1
    df_trt['Publication number'] = pubno1
    df_trt['T1'] = t1
    df_trt['prep'] = prep
    df_trt['T2'] = t2
    # df_trt.to_csv('TRTs.csv',index=False)
    return df_trt



