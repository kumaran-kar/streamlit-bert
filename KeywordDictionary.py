import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#import PatentInfo

def getkeywords(abstracts,min_word,max_word,min_df = 0.01,max_df=0.99):
    # abstracts = PatentInfo.get_abstract(patent_path)
    vectorizer = TfidfVectorizer(ngram_range=(min_word,max_word),lowercase=True,min_df = min_df,max_df = max_df,stop_words='english') 
    tfidf_matrix = vectorizer.fit_transform(abstracts)
    sum_scores = tfidf_matrix.sum(axis=0)
    keywords = [(term,sum_scores[0,idx]) for term,idx in vectorizer.vocabulary_.items()]
    keywords.sort(key = lambda x: x[1], reverse=True)
    # for k in keywords:
    #     kws.append(k[0])
    return keywords



def getkeyword_dict(abstracts,threshold,min_word,max_word,nlp):
    keys = getkeywords(abstracts,min_word,max_word)
    kws = []
    for k in keys:
        kws.append(k[0])
    #nlp = spacy.load(r"en_core_web_trf-3.5.0")
    kw_vectors = [nlp(keyword).vector for keyword in kws]
    similarity_matrix = np.array(cosine_similarity(kw_vectors,kw_vectors))
    keyword_dict = dict()
    # jindex = list()
    # for i in range(0,len(similarity_matrix)):
    #     temp = list()
    #     if i not in jindex:
    #         for j in range(0,len(similarity_matrix)):
    #             if similarity_matrix[i][j] >= 0.80:
    #                 temp.append(kws[j]) 
    #                 jindex.append(j)
    #         keyword_dict.update({kws[i]:temp})
    #     else:
    #         continue

    jindex = set()
    for i in range(len(similarity_matrix)):
        if i not in jindex:
            temp = [kws[j] for j in np.where(similarity_matrix[i] >= threshold)[0] if j not in jindex]
            keyword_dict[kws[i]] = temp
            jindex.update(np.where(similarity_matrix[i] >= threshold)[0])
    return keyword_dict 