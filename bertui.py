import streamlit as st
import pandas as pd
import numpy as np
import PatentInfo
import GenerateTRT
import mpnet_load
import TextEmbedding
import keywordBERT
# import Clustering
import KeywordMapping

st.set_page_config(initial_sidebar_state='collapsed',layout='wide')


@st.cache_data
def abstract_pubno(file):
    abstracts = PatentInfo.get_abstract(file)
    pubno = PatentInfo.get_pubno(file)
    return {'abstracts':abstracts,'pubno':pubno}

@st.cache_resource
def load_spacy():
    import spacy
    nlp = spacy.load('en_core_web_md')
    return nlp

@st.cache_resource
def load_model():
    model = mpnet_load.load_mpnet()
    return model

def countwords(string):
    words = string.split()
    return len(words)

@st.cache_data
def Keyphrases(abstracts):
    nlp = load_spacy()
    from keyphrase_vectorizers import KeyphraseTfidfVectorizer
    vectorizer = KeyphraseTfidfVectorizer(spacy_pipeline=nlp)
    vectorizer.fit(abstracts)
    keyphrases = vectorizer.get_feature_names_out()
    filterphrases = list()
    for k in keyphrases:
        if not countwords(k) <= 1 :
            filterphrases.append(k)
    st.write(filterphrases)
    return filterphrases


@st.cache_data
def embeddings(abstracts,min_word,max_word,stopwords,candidates):
    kw_model = load_model()
    embedding = TextEmbedding.mpnet_embedding(kw_model,abstracts,min_word,max_word,
                                              stopwords=stopwords,candidates=candidates)
    return embedding

@st.cache_data
def TRTdf(abstracts,pubno):
    nlp = load_spacy()
    df_trt = GenerateTRT.TRTdf(abstracts,pubno,nlp)
    return df_trt

@st.cache_data
def keywordsTFIDF(abstracts,min_word,max_word,stopwords,min_df=0.01,max_df=0.99):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(min_word,max_word),lowercase=True,min_df = min_df,max_df = max_df,stop_words=stopwords) 
    tfidf_matrix = vectorizer.fit_transform(abstracts)
    sum_scores = tfidf_matrix.sum(axis=0)
    keywords = [(term,sum_scores[0,idx]) for term,idx in vectorizer.vocabulary_.items()]
    keywords.sort(key = lambda x: x[1], reverse=True)
    kws = []
    for k in keywords:
        kws.append(k[0])
    return kws

@st.cache_data
def keywords_mpnet(abstracts,embedding,min_word,max_word,no_keywords,stopwords,candidates,threshold=None):
    kw_model = load_model()
    keywords = keywordBERT.KeywordBERT(abstracts,kw_model,embedding,min_word,max_word,no_keywords,stopwords=stopwords,candidates=candidates,threshold=threshold)
    # kws = []
    # for k in keywords:
    #     kws.append(k[0])
    return keywords

@st.cache_data
def mapkeywords(df,keywords):
    df_map = KeywordMapping.keywordmap(df,keywords)
    return df_map

# @st.cache_data
# def clustering(abstracts_embedded,pubno,abstracts):
#     df = Clustering.cluster_abstracts(abstracts_embedded,pubno,abstracts)
#     return df

@st.cache_data
def keywordsynonyms(kws,threshold):
    from sklearn.metrics.pairwise import cosine_similarity
    nlp = load_model()
    kw_vectors,dummyvecs = nlp.extract_embeddings(kws,keyphrase_ngram_range=(3,3),stop_words='english')
    # kw_vectors = [nlp(keyword).vector for keyword in kws]
    similarity_matrix = np.array(cosine_similarity(kw_vectors,kw_vectors))
    keyword_dict = dict()
    jindex = set()
    for i in range(len(similarity_matrix)):
        if i not in jindex:
            temp = [kws[j] for j in np.where(similarity_matrix[i] >= threshold)[0] if j not in jindex]
            keyword_dict[kws[i]] = list(set(temp))
            jindex.update(np.where(similarity_matrix[i] >= threshold)[0])   
    return keyword_dict 

@st.cache_data
def mapkeydict(df,keywords,keydict):
    dfmap = KeywordMapping.keydictmap(df,keywords,keydict)
    return dfmap

@st.cache_data
def get_keywords_df(df):
    kws = set(list(df['T1'])) | set(list(df['T2']))
    return list(kws)

# @st.cache_data
# def graph_data(df,keystouse):
#     nodes = []
#     edges = []
    
#     for index,row in df.iterrows():
#         if (row['T1'] and row['T2']) in keystouse:
#             if len(df['Prep'].unique()) > 1:
#                 edges.append(Edge(source=row['T1'],target=row['T2'],color=row['Prep']))
#             else:
#                 edges.append(Edge(source=row['T1'],target=row['T2']))
#     kws = get_keywords_df(df)
#     for kw in kws:
#         if kw in keystouse:
#             # pubno = 
            
#             nodes.append(Node(id = kw,size=10))
#     config = Config(width=750,
#                 height=950,
#                 directed=True, 
#                 physics=False, 
#                 hierarchical=False)
#     return nodes,edges,config

@st.cache_data
def jsonlist(df,keywords_scores,keystouse):
    tempdict = dict()
    t1list = list(df['T1'].unique())
    # preplist = list(df['Prep'].unique())
    for t1 in t1list:
        if t1 in keystouse:
            temp = df[df['T1']==t1]
            t2s = set(temp['T2'])
            t3s = [i for i in t2s if i in keystouse]
            df1 = df[df['T1'].isin([t1])]
            templist = list()
            for t in t3s:
                df2 = df1[df1['T2'].isin([t])]
                pubno = list(df2['Pubno.'])
                templist.append(t)


            
            tempdict.update({t1:templist})
    # sorted_dict = dict(sorted(tempdict.items(),key = lambda x: [t[1] for t in keywords_scores if t[0] == x[0][0]],reverse=True))
    return tempdict

def pyvisgraph(jsont):
    from pyvis.network import Network
    net = Network(filter_menu=True,neighborhood_highlight=True,directed=True)
    import networkx as nx
    g = nx.Graph()
    nodes = list(jsont.keys())
    for k in jsont.keys():
        for v in jsont[k]:
            if v not in nodes:
                nodes.append(v)
    g.add_nodes_from(nodes)
    for k in jsont.keys():
        for v in jsont[k]:
            g.add_edge(k,v)
    net.from_nx(g)
    net.toggle_physics(True)
    net.toggle_drag_nodes(True)
    net.show_buttons()
    net.toggle_stabilization(status= True)
    net.save_graph('graph.html')


# def networkxgraph(df,nodestouse):
@st.cache_data
def prominenttech(jsont):
    import networkx as nx
    st.markdown("Prominent nodes are found out by their respective Degree Centrality scores")
    
    g = nx.Graph()
    nodes = list(jsont.keys())
    for k in jsont.keys():
        for v in jsont[k]:
            if v not in nodes:
                nodes.append(v)
    g.add_nodes_from(nodes)
    for k in jsont.keys():
        for v in jsont[k]:
            g.add_edge(k,v)
    dict = nx.degree_centrality(g)
    st.write(dict)
    
    # sorted_dict = dict(sorted(promidict.keys(),reverse=True))


@st.cache_data
def techoutliers(df,patentinfo,relationship,keystouse):
    st.markdown("Detecting Technology outliers")
    from keyphrase_vectorizers import KeyphraseTfidfVectorizer
    vect = KeyphraseTfidfVectorizer(spacy_pipeline=r'en_core_web_lg-3.5.0',stop_words='english',max_df=1)
    vect.fit(patentinfo['abstracts'])
    templ = list(vect.get_feature_names_out())
    # from sklearn.metrics.pairwise import cosine_similarity

    # dfmap = KeywordMapping.keywordmap(df,templ)
    # df1 = dfmap[dfmap['Prep']==relationship]
    # tempdict = dict()
    # t1list = list(df1['T1'].unique())
    st.write(templ)
    # keydict = keywordsynonyms()
    # preplist = list(df['Prep'].unique())
    # for t1 in t1list:
    #     temp = df1[df1['T1']==t1]
    #     t2s = set(temp['T2'])
    #     df1 = df[df['T1'].isin([t1])]
    #     templist = list()
    #     for t in t2s:
    #         df2 = df1[df1['T2'].isin([t])]
    #         pubno = list(df2['Pubno.'])
    #         templist.append({t:pubno})
    #         tempdict.update({t1:templist})
    # sorted_dict = dict(sorted(tempdict.items(),key = lambda x: [t[1] for t in keywords_scores if t[0] == x[0][0]],reverse=True))
    # return tempdict



def main():  ### tfidf better ig
    st.header("Technology Relationhip Extraction using BERT-like MPNet")
    st.markdown('Graph network depicting technology terms and their relationship')
    file = st.file_uploader("Upload xlsx file",type=['xlsx'])
    
    if file is not None:
        df = pd.read_excel(file)
        patentinfo = abstract_pubno(file)
        # st.write('Change parameters in sidebar(Only if results are substandard!! Will increase computational time..)')

        
        # min_word = st.sidebar.selectbox("Minimum no of words in keyword(Ideal - 2):",[2,3],index = 0)
        # max_word = st.sidebar.selectbox("Maximum no of words in keyword(Ideal - 3):",[2,3,4],index=1)
        min_word = 2
        max_word = 4
        # # diversity = st.sidebar.select_slider("Pick diversity threshold",options=range(60,75,5), value=65)
        # # relationship = st.sidebar.selectbox("Select Relationship type",['Inclusion','Objective','Effect','Process','Likeness','Misc'])
        # stopwords = st.sidebar.selectbox('Want to remove stopwords???',['Yes','No'],index = 0)
        # candidate = st.sidebar.selectbox('Want to use candidate keywords???',['Yes','No'],index=0)
            
        # if stopwords == 'Yes':
        #     stopwords = 'english'
        # else:
        #     stopwords = None
        stopwords = 'english'
        df_trt = TRTdf(patentinfo['abstracts'],patentinfo['pubno'])
        
        # if candidate == 'Yes':
        candidate_keys = Keyphrases(patentinfo['abstracts'])
        # candidate_keys = None
        # else:
        #     candidate_keys = None
        tab1,tab2 , tab3= st.tabs(['Graph Generation','Technological outliers','Graph Analytics'])
        with st.spinner('Abstracts embedding in progress....'):
                embedding = embeddings(patentinfo['abstracts'],min_word,max_word,stopwords=stopwords,candidates=candidate_keys)
        
        with tab1:

            
            relationship = st.selectbox("Select Relationship type",['Inclusion','Objective','Effect','Process','Likeness','Misc'])
            threshold = st.select_slider("Pick similarity threshold",options=range(70,95,5), value=80)
            no_keywords = st.select_slider("No of keywords extracted from a single abstract",options=range(5,11,2),value=7)
                
            
            with st.container():        
                keywords_total = keywords_mpnet(patentinfo['abstracts'],embedding,min_word,max_word,no_keywords,stopwords,candidates = candidate_keys,threshold=None)
                keywords_scores=list()
                keywords= []
                for k in keywords_total:
                    keywords_scores.append(k)
                    for k2 in k:
                        keywords.append(k2[0])
                keydict = keywordsynonyms(keywords,threshold*0.01)
                # c1 = "hydrogen leakage rate measurement method"
                # c2 = "lowtemperature hydrogen leakage detection performance"
                # m = load_model()
                # l = [c1,c2]
                # em,tm = m.extract_embeddings(l,keyphrase_ngram_range=(3,3),stop_words='english')
                # from sklearn.metrics.pairwise import cosine_similarity
                # st.write(cosine_similarity(em,em))

                st.json(keydict,expanded=False)
                dfmap = mapkeydict(df_trt,keywords,keydict)
                df_keywords = get_keywords_df(dfmap[dfmap['Prep']==relationship])
                with st.container():
                        
                    keystouse = st.multiselect("De-select irrelevant technology terms",df_keywords,default=df_keywords)
                        # submitted1 = st.form_submit_button(label='Submit')
                    # submitted2 = st.form_submit_button(label='Submit')
                    
                with st.container():
                            
                    df_temp = dfmap[dfmap['Prep']==relationship]
                    jsont = jsonlist(df_temp,keywords_scores,keystouse)
                    st.write(len(jsont))
                    st.json(jsont,expanded=False)
                    
                        # dict_display()
                        
                with st.container():
                    with st.form("form1"):
                        nodestouse = st.multiselect("Select nodes to construct Graph Network",keystouse,default=keystouse)
                        submitted = st.form_submit_button(label='Submit')
                            
                    if submitted:
                        from pyvis.network import Network
                        pyvisgraph(jsont)
                        # agraph(nodes=nodes,edges=edges,config=config)
                        # igraphgen(dfmap[dfmap['Prep']==relationship],nodestouse)

                        with open(r'graph.html', 'r') as f:
                            html_string = f.read()
                        st.download_button(label='Download Graph Network in .html format',file_name=r'graph.html',data=html_string)

        with tab2:
            # relationship1 = st.selectbox("Select Relationship type",['Inclusion','Objective','Effect','Process','Likeness','Misc'],key=9,)
            relationship1 = 'Inclusion'
            inc = ['of','in','with','from','on','at','within','includes','by'] ## Inclusion
            obj = ['for'] ## Objective
            eff = ['to','across','against'] ## Effect
            pro = ['during','into','through','via'] ## Process
            like = ['as'] ## Likeness
            techoutliers(df_trt,patentinfo,relationship1,keystouse)
            # st.json(outlierdict)
            
        with tab3:   
            prominenttech(jsont)
            



        #     st.write("temp")
            # with st.spinner('Clustering the abstracts'):
                
            #     df_cluster = clustering(embedding[0],patentinfo['abstracts'],patentinfo['pubno'])
            
            # # relationship = st.selectbox("Select Relationship type",['Inclusion','Objective','Effect','Process','Likeness','Misc'],key=1)
            # diversity = st.select_slider("Pick diversity threshold",options=range(55,75,5), value=60,key =2)

            # import matplotlib.pyplot as plt
            # df_cluster.sort_values(by='Labels',inplace=True,axis=0)
            # clusters = df_cluster['Labels'].unique()
            # cluster = None
            # st.write("Cluster count:")
            # st.dataframe(df_cluster['Labels'].value_counts())
            # cluster = st.selectbox('Select cluster',clusters,key = 3)
            # if not cluster is None:
            #     df1 = df_cluster[df_cluster['Labels']== cluster]
            #     pubnodf1 = list(df1['Pubno'])
            #     df_clustertrt = df_trt[df_trt['Publication number'].isin(pubnodf1)]
            #     abstracts_cluster = list(df1['Abstract'])
                
            #     candidatekeys_cluster = keywordsTFIDF(abstracts_cluster,2,3,stopwords='english')
            #     # if candidate == 'Yes':
            #     #     candidate_keys = keywordsTFIDF(abstracts_cluster,min_word,max_word,stopwords=stopwords)
            #     # else:
            #     #     candidate_keys = None
            #     with st.spinner("Embedding in progress.."):
            #         embedding_cluster = embeddings(abstracts_cluster,min_word,max_word,stopwords='english',candidates=None)
            #     with st.container():        
            #         keywords_scores_cluster = keywords_mpnet(abstracts_cluster,embedding_cluster,min_word,max_word,len(df_clustertrt)*2,stopwords='english',candidates=None,threshold= diversity*0.01)
            #         keywords_cluster=[]
            #         for k in keywords_scores_cluster:
            #             keywords_cluster.append(k[0])
            #         # keydict_cluster = keywordsynonyms(keywords_cluster,threshold)
            #         # dfmap_cluster = mapkeydict(df_clustertrt,keywords_cluster,keydict_cluster)
            #         dfmap_cluster = KeywordMapping.keywordmap(df_clustertrt,keywords_cluster)
            #         st.write(dfmap_cluster)
            #         df_keywords_cluster = get_keywords_df(dfmap_cluster)
                    
            #         with st.container():

            #             keystouse1 = st.multiselect("De-select irrelevant technology terms",df_keywords_cluster,default=df_keywords_cluster,key=5)
            #         with st.container():
            #             df_temp1 = dfmap_cluster
            #             jsont1 = jsonlist(df_temp1,keywords_scores_cluster,keystouse1)
            #             st.json(jsont1,expanded=False)
            #         with st.container():
            #             nodes,edges,config= graph_data(dfmap_cluster,keystouse1)
            #             agraph(nodes=nodes,edges=edges,config=config)


if __name__ == "__main__":
    main()







                
                
                
                



        



