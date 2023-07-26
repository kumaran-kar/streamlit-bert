from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer




def mpnet_embedding(kw_model,abstracts,min_word,max_word,stopwords = 'english',candidates = None):
    embedding = kw_model.extract_embeddings(abstracts,keyphrase_ngram_range=(min_word,max_word),
                                                                    stop_words=stopwords,candidates=candidates)
                                                                    # vectorizer = KeyphraseCountVectorizer(spacy_pipeline=r'en_core_web_lg-3.5.0')
                                                                    
    return embedding
    