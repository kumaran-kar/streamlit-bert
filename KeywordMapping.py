import re
import pandas as pd

def keywordmap(df,listkw):
    pattern = fr'\b{"|".join(listkw)}\b'
    t1 = df['T1'].str.findall(pattern,flags = re.IGNORECASE).tolist()
    t2 = df['T2'].str.findall(pattern, flags = re.IGNORECASE).tolist()
    pubno = df['Publication number']
    prep = df['prep']
    dfmap2 = pd.DataFrame()
    dfmap2['Pubno.'] = pubno
    dfmap2['T1'] = t1
    dfmap2['Prep'] = prep
    dfmap2['T2'] = t2
    
    t1 = list()
    t2 = list()
    prep = list()
    pubno = list()
    inc = ['of','in','with','from','on','at','within','includes','by'] ## Inclusion
    obj = ['for'] ## Objective
    eff = ['to','across','against'] ## Effect
    pro = ['during','into','through','via'] ## Process
    like = ['as'] ## Likeness
    for index,row in dfmap2.iterrows():
        
        if row['T1']:
            for rowt1 in row['T1']:
                if row['T2']:
                    for rowt2 in row['T2']:
                        t1.append(rowt1)
                        t2.append(rowt2)
                        pubno.append(row['Pubno.'])
                        prep.append(row['Prep'])
        
    dfmap3 = pd.DataFrame()
    dfmap3['T1'] = t1
    dfmap3['T2'] = t2
    dfmap3['Prep'] = prep
    dfmap3['Pubno.'] = pubno
    for index,row in dfmap3.iterrows():
        if row['Prep'] in inc:
            dfmap3.at[index,'Prep'] = 'Inclusion'
        elif row['Prep'] in obj:
            dfmap3.at[index,'Prep'] = 'Objective'
        elif row['Prep'] in eff:
            dfmap3.at[index,'Prep'] = 'Effect'
        elif row['Prep'] in pro:
            dfmap3.at[index,'Prep'] = 'Process'
        elif row['Prep'] in like:
            dfmap3.at[index,'Prep'] = 'Likeness'
        else:
            dfmap3.at[index,'Prep'] = 'Misc'

    return dfmap3

def keydictmap(df,listkw,keydict):
    dfmap = keywordmap(df,listkw)
    # pubno = list()
    # t1 = list()
    # prep = list()
    # t2 = list()
    # for index,row in dfmap.iterrows():
    #     if row['T1']:
    #         for rowt1 in row['T1']:
    #             if row['T2']:
    #                 for rowt2 in row['T2']:
    #                     t1.append(rowt1)
    #                     t2.append(rowt2)
    #                     pubno.append(row['Pubno.'])
    #                     prep.append(row['Prep'])
    # dfmap2 = pd.DataFrame()
    # dfmap2['T1'] = t1
    # dfmap2['T2'] = t2
    # dfmap2['Prep'] = prep
    # dfmap2['Pubno.'] = pubno
    # dfmap2.to_csv("TRT2.csv",index=False)
    
    
    for index,row in dfmap.iterrows():
        t1 = row['T1']
        t2 = row['T2']
        for key, value in keydict.items():
            if t1 in value:
                dfmap.at[index,'T1'] = key
            if t2 in value:
                dfmap.at[index,'T2'] = key
            

    return dfmap

