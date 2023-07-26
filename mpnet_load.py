from keybert import KeyBERT

def load_mpnet():
    kw_model = KeyBERT(model=r'mp_net')
    return kw_model