import pandas as pd
import TRTExtraction as trtext
import PatentInfo as ptinfo
from ast import literal_eval


def get_trt(abstract,nlp):  ## Finds trts in a given text document and returns the trt in list of trt dictionaries
    
    trt = list()
    pplist = ['ADP']
    doc = nlp(abstract)
    
    sentences = list(doc.sents)
    for sentence in sentences:
        token_iter = 0
        token_pos = [token.pos_ for token in sentence]
        token_text = [token.text for token in sentence]
        for token in sentence:
            if token.pos_ in pplist:
                token_iter1 = token_iter-1
                t1 = " "
                while token_pos[token_iter1] not in pplist:
                    temp = t1
                    if temp == " ":
                        t1 = token_text[token_iter1]
                    else:
                        t1 = token_text[token_iter1]+" "+temp
                    if not token_iter1 == 0:
                        token_iter1 = token_iter1 -1
                    else:
                        break

                token_iter2 = token_iter+1
                t2 = " "
                while token_pos[token_iter2] not in pplist:
                    temp = t2
                    if temp == " ":
                        t2 = token_text[token_iter2]
                    else:
                        t2 = temp+" "+token_text[token_iter2]
                    if not token_iter2 == len(token_text)-1:
                        token_iter2 = token_iter2 + 1
                    else:
                        break
                temp = {"T1":t1, "Preposition":token.text, "T2":t2}
                trt.append(temp)
                
            token_iter = token_iter + 1
    return trt

# def main(patent_path):
#     abstracts = ptinfo.get_abstract(patent_path)
#     pubno = ptinfo.get_pubno(patent_path)
#     ##df = pd.DataFrame()
#     ##df['Publication number'] = pubno
#     ##df['Abstarct'] = abstracts
#     trts = list()
#     nlp = spacy.load(r"en_core_web_md-3.5.0")
#     for abstract in abstracts:
#         trt = trtext.get_trt(abstract,nlp)
#         trts.append(trt)
#     ##df['TRT'] = trts
#     df_trt = pd.DataFrame()
#     pubno1 = list()
#     t1 = list()
#     prep = list()
#     t2 = list()
#     iter = 0
#     for row in trts :
#         for trt in row:
#             pubno1.append(pubno[iter])
#             t1.append(trt['T1'])
#             prep.append(trt['Preposition'])
#             t2.append(trt['T2'])
#         iter = iter + 1
#     df_trt['Publication number'] = pubno1
#     df_trt['T1'] = t1
#     df_trt['prep'] = prep
#     df_trt['T2'] = t2
#     df_trt.to_csv('TRTs.csv',index=False)

# if __name__ == "__main__":
#     main(r"Hydrogen leak.xlsx")

#  for sentence in doc.sents:
#         for token in sentence:
#             if token.pos_ in pplist:
#                 token_index = token.i
#                 t1 = " ".join(token_text for token_text in (t.text for t in sentence[:token_index] if t.pos_ not in pplist)
#                 t2 = " ".join(token_text for token_text in (t.text for t in sentence[token_index+1:] if t.pos_ not in pplist)