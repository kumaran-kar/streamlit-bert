from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseTfidfVectorizer

def KeywordBERT(abstracts,kw_model,embedding,min_word,max_word,no_keywords,stopwords,candidates=None,threshold=None):
    if threshold == None:
        keyword_scores = kw_model.extract_keywords(docs = abstracts,doc_embeddings=embedding[0],word_embeddings=embedding[1],
                                               keyphrase_ngram_range=(min_word,max_word),stop_words=stopwords, top_n= no_keywords,candidates=candidates)
                                            #     vectorizer = KeyphraseTfidfVectorizer(spacy_pipeline=r'en_core_web_lg-3.5.0'))
    else:
        keyword_scores = kw_model.extract_keywords(docs = abstracts,doc_embeddings=embedding[0],word_embeddings=embedding[1],
                                               stop_words=stopwords, keyphrase_ngram_range=(min_word,max_word),
                                               top_n= no_keywords,use_mmr=True,diversity=threshold,vectorizer = KeyphraseTfidfVectorizer(spacy_pipeline=r'en_core_web_lg-3.5.0'))
        

    # keywords = []
    # for k in keyword_scores:
    #     keywords.append(k[0])
    
    return keyword_scores



    