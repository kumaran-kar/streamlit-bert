import pandas as pd
import re

def clean_text(text):
    cleaned_text = re.sub(r"\(.*\)\n","" ,text) ## Removes (<somecharacters>)\n
    cleaned_text = cleaned_text.replace(';','.') ## Replaces ; with .
    cleaned_text = cleaned_text.replace('\n',' ') ## Replaces \n with " "
    cleaned_text = cleaned_text.replace('-'," ")
    cleaned_text = cleaned_text.replace(","," ") 
    cleaned_text1 = re.sub(r'\bPROBLEM TO BE SOLVED:\b',"",cleaned_text)
    # cleaned_text2 = re.sub(r'\b:\b',"",cleaned_text)
    return cleaned_text1

def get_abstract(patent_path): ## Cleans abstract and returns abstract
    df = pd.read_excel(patent_path)
    df.dropna(subset=['Abstract','Publication numbers'],axis=0,inplace=True)
    abstract = list(df['Abstract'])
    cleaned_abs = list(map(clean_text,abstract))
    return cleaned_abs

def get_pubno(patent_path):
    df = pd.read_excel(patent_path)
    df.dropna(subset=['Abstract','Publication numbers'],axis=0,inplace=True)
    pubno = list(df['Publication numbers'])
    return pubno



    

