from keybert import KeyBERT

def load_mpnet():
    kw_model = KeyBERT(model=r'all-mpnet-base-v2')
    return kw_model
