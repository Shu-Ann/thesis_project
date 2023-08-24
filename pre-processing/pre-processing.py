import pandas as pd
import re

# --------------- read files --------------- 
pepper_1 = pd.read_csv('./data/Pepper_1.csv')
pepper_2 = pd.read_csv('./data/Pepper_2.csv')
pepper_3 = pd.read_csv('./data/Pepper_3.csv')
pepper_4 = pd.read_csv('./data/Pepper_4.csv')

fetch_1 = pd.read_csv('./data/Fetch_1.csv')
fetch_2 = pd.read_csv('./data/Fetch_2.csv')
fetch_3 = pd.read_csv('./data/Fetch_3.csv')
fetch_4 = pd.read_csv('./data/Fetch_4.csv')

# -------------- columns drop & rename -------------------
pepper_1=pepper_1.drop(['Event'], axis=1)
pepper_1 = pepper_1.rename(columns={'Duration - ss.msec': 'Duration', 'Begin Time - ss.msec':'Start', 'End Time - ss.msec':'End'})

pepper_2=pepper_2.drop(['Event'], axis=1)
pepper_2 = pepper_2.rename(columns={'Duration - ss.msec': 'Duration', 'Begin Time - PAL': 'Start', 'End Time - PAL':'End' })

pepper_3=pepper_3.drop(['Begin Time - hh:mm:ss.ms',
                        'End Time - hh:mm:ss.ms','Duration - hh:mm:ss.ms','Event'], axis=1)
pepper_3 = pepper_3.rename(columns={'Duration - ss.msec': 'Duration', 'Begin Time - ss.msec':'Start', 'End Time - ss.msec':'End'})

pepper_4=pepper_4.drop(['SentenceID',"Begin Time - hh:mm:ss.ms",'End Time - hh:mm:ss.ms',
                        'Duration - hh:mm:ss.ms','Event'], axis=1)
pepper_4 = pepper_4.rename(columns={'Duration - ss.msec': 'Duration', 'Begin Time - ss.msec':'Start', 'End Time - ss.msec':'End'})

fetch_1=fetch_1.drop(['Tier-0','Tier-4'], axis=1)
fetch_1 = fetch_1.rename(columns={'Duration - ss.msec': 'Duration', 'Tier-2':'comment', 'Tier-1':'Role',
                                  'Tier-3':'Label', 'Begin Time - ss.msec':'Start', 'End Time - ss.msec':'End'})
 
fetch_2=fetch_2.drop(['Event'], axis=1)
fetch_2=fetch_2.rename(columns={'Begin Time - hh:mm:ss.ms':'Start', 'End Time - hh:mm:ss.ms':'End'})

fetch_3=fetch_3.rename(columns={'Begin Time - hh:mm:ss.ms':'Start', 'End Time - hh:mm:ss.ms':'End'})

fetch_4=fetch_4.rename(columns={'Begin Time - hh:mm:ss.ms':'Start', 'End Time - hh:mm:ss.ms':'End'})

# ------------- time duration ------------------
def duration(df):
    df['End'] = df['End'].apply(lambda x: int(x[:8].split(':')[0])*3600+int(x[:8].split(':')[1])*60+int(x[:8].split(':')[2]))
    df['Start'] = df['Start'].apply(lambda x: int(x[:8].split(':')[0])*3600+int(x[:8].split(':')[1])*60+int(x[:8].split(':')[2]))
    df['Duration']='0'
    for row in range(0, len(df)):
        df['Duration'][row] = (int(df['End'][row])-int(df['Start'][row]))
        
        
duration(pepper_2)
duration(fetch_2)
duration(fetch_3)
duration(fetch_4)

# ------------ remove fillers -------------------
pepper_1 = pepper_1[(pepper_1['Label']!='<fillers>') & (pepper_1['Label']!='<robot speech>')]
pepper_1 = pepper_1.reset_index(drop=True)

pepper_2 = pepper_2[(pepper_2['Label']!='<fillers>') & (pepper_2['Label']!='<robot speech>')]
pepper_2 = pepper_2.reset_index(drop=True)

pepper_3 = pepper_3[(pepper_3['Label']!='<fillers>') & (pepper_3['Label']!='<robot speech>')]
pepper_3 = pepper_3.reset_index(drop=True)

pepper_4 = pepper_4[(pepper_4['Label']!='<fillers>') & (pepper_4['Label']!='<robot speech>')]
pepper_4 = pepper_4.reset_index(drop=True)

fetch_1 = fetch_1[(fetch_1['Label']!='<fillers>') & (fetch_1['Label']!='<robot speech>')]
fetch_1 = fetch_1.reset_index(drop=True)

fetch_2 = fetch_2[(fetch_2['Label']!='<fillers>') & (fetch_2['Label']!='<robot speech>')]
fetch_2 = fetch_2.reset_index(drop=True)

fetch_3 = fetch_3[(fetch_3['Label']!='<fillers>') & (fetch_3['Label']!='<robot speech>')]
fetch_3 = fetch_3.reset_index(drop=True)

fetch_4 = fetch_4[(fetch_4['Label']!='<fillers>') & (fetch_4['Label']!='<robot speech>')]
fetch_4 = fetch_4.reset_index(drop=True)


# ----------------  merge rows -------------------
class merge:
    
    def __init__(self, df, de_rows):
        self.df=df
        self.de_rows=[]

    def combine(self):
        row=1
        self.de_rows=[]
        self.df['combine']='0'
        while row<len(self.df):
            if (self.df['Role'][row]==self.df['Role'][row-1]) and (self.df['Label'][row]==self.df['Label'][row-1]) and row<2 and int(self.df['Duration'][row-1])<=5:
                self.df['combine'][row]=self.df['comment'][row-1]+' '+self.df['comment'][row]
                self.df['Start'][row]=self.df['Start'][row-1]
            elif (self.df['Role'][row]==self.df['Role'][row-1]) and (self.df['Label'][row]==self.df['Label'][row-1]) and row>=2 and int(self.df['Duration'][row-1])<=5:
                self.df['combine'][row]=self.df['combine'][row-1]+' '+self.df['comment'][row]
                self.de_rows.append(row-1)
                self.df['Start'][row]=self.df['Start'][row-1]
            else:
                self.df['combine'][row]=self.df['comment'][row]

            row+=1


        self.df['combine'][0]=self.df['comment'][0]

        if (self.df['Role'][0]==self.df['Role'][1]) and (self.df['Label'][0]==self.df['Label'][1])and int(self.df['Duration'][0])<=5:
            self.de_rows.append(0)


    def drop(self):
        self.df=self.df.drop(index=self.de_rows)

        return self.df
    

A=merge(pepper_1,[])
A.combine()
pepper_1=A.drop()

B=merge(pepper_2,[])
B.combine()
pepper_2=B.drop()

C=merge(pepper_3,[])
C.combine()
pepper_3=C.drop()

D=merge(pepper_4,[])
D.combine()
pepper_4=D.drop()

E=merge(fetch_1,[])
E.combine()
fetch_1=E.drop()

F=merge(fetch_2,[])
F.combine()
fetch_2=F.drop()

G=merge(fetch_3,[])
G.combine()
fetch_3=G.drop()

H=merge(fetch_4,[])
H.combine()
fetch_4=H.drop()

# ------------- replace symbols (labels) --------------------
def replace_sym(df):
    df['Label']=df['Label'].apply(lambda x: str(x).replace('<','').replace('>',''))

replace_sym(pepper_1)
replace_sym(pepper_2)
replace_sym(pepper_3)
replace_sym(pepper_4)

replace_sym(fetch_1)
replace_sym(fetch_2)
replace_sym(fetch_3)
replace_sym(fetch_4)

# ------------- replace symbols in text ---------------
def text_preprocessing(data):
    data = re.sub(r'(@.*?)[\s]', ' ', data)
    data = re.sub(r'[0-9]+' , '' ,data)
    data = re.sub(r'\s([@][\w_-]+)', '', data).strip()
    data = re.sub(r'&amp;', '&', data)
    data = re.sub(r'\s+', ' ', data).strip()
    data = data.replace("#" , " ")
    data = data.replace('"' , " ")
    data = data.lower()
    encoded_string = data.encode("ascii", "ignore")
    decode_string = encoded_string.decode()
    return decode_string

pepper_1['text']=pepper_1['combine'].apply(text_preprocessing)
pepper_2['text']=pepper_2['combine'].apply(text_preprocessing)
pepper_3['text']=pepper_3['combine'].apply(text_preprocessing)
pepper_4['text']=pepper_4['combine'].apply(text_preprocessing)

fetch_1['text']=fetch_1['combine'].apply(text_preprocessing)
fetch_2['text']=fetch_2['combine'].apply(text_preprocessing)
fetch_3['text']=fetch_3['combine'].apply(text_preprocessing)
fetch_4['text']=fetch_4['combine'].apply(text_preprocessing)

# ---------- Role ----------
pepper_1_R=pepper_1[pepper_1['Role']=='R']
pepper_1_P=pepper_1[pepper_1['Role']=='P']

pepper_2_R=pepper_2[pepper_2['Role']=='R']
pepper_2_P=pepper_2[pepper_2['Role']=='P']

pepper_3_R=pepper_3[pepper_3['Role']=='R']
pepper_3_P=pepper_3[pepper_3['Role']=='P']

pepper_4_R=pepper_4[pepper_4['Role']=='R']
pepper_4_P=pepper_4[pepper_4['Role']=='P']

fetch_1_R=fetch_1[fetch_1['Role']=='R']
fetch_1_P=fetch_1[fetch_1['Role']=='P']

fetch_2_R=fetch_2[fetch_2['Role']=='R']
fetch_2_P=fetch_2[fetch_2['Role']=='P']

fetch_3_R=fetch_3[fetch_3['Role']=='R']
fetch_3_P=fetch_3[fetch_3['Role']=='P']

fetch_4_R=fetch_4[fetch_4['Role']=='R']
fetch_4_P=fetch_4[fetch_4['Role']=='P']

# --------------main label----------------
def exclude(df, labels):
    exclude_idx=[]
    for d in range(0, len(df['Label'])):
        if df['Label'][d] not in labels:
            exclude_idx.append(d)
    return df.drop(index=(exclude_idx))

R_sublabels=['introduce background','prepare demonstration', 'provide clarification','additional info',
          'explain behavior','explain resources',
          'refer to resources','refer to simulation',
          'prompt action','prompt clarification','prompt evaluation',
          'confirm intention','provide opinion',
          'summarize discussion','time management','encouragement',
          'prompt resources clarification', 'propose action','propose approximation',
          'propose choice','implement behavior', 'identify failure','explain failure',
          'debugging','identify limitation',
          'robot limitation','resources - setup limitation']


P_sublabels=['ask for clarification','accept suggestion','accept clarification','call for discussion',
          'propose role','propose behavior',
          'choose behavior','explain proposed behavior',
          'clarification reasoning','refer to experience', 'propose action',
          'propose replacement','propose fixes',
          'propose addition','propose removal',
          'identify failure', 'identify limitation',
          'social context','spatial context','user context','liability concern',
          'safety concern','ethical concern',
          'robot limitation', 'resources - setup limitation',
          'positive','indifferent','anthropomorphize','unsuitable goal',
          'interaction - engagement failure','performance failure',
          'inappropriate behavior','unexpected behavior', 'refer to simulation']

pepper_1_R = pepper_1_R.reset_index(drop=True)
pepper_2_R = pepper_2_R.reset_index(drop=True)
pepper_3_R = pepper_3_R.reset_index(drop=True)
pepper_4_R = pepper_4_R.reset_index(drop=True)

fetch_1_R = fetch_1_R.reset_index(drop=True)
fetch_2_R = fetch_2_R.reset_index(drop=True)
fetch_3_R = fetch_3_R.reset_index(drop=True)
fetch_4_R = fetch_4_R.reset_index(drop=True)

pepper_1_P = pepper_1_P.reset_index(drop=True)
pepper_2_P = pepper_2_P.reset_index(drop=True)
pepper_3_P = pepper_3_P.reset_index(drop=True)
pepper_4_P = pepper_4_P.reset_index(drop=True)

fetch_1_P = fetch_1_P.reset_index(drop=True)
fetch_2_P = fetch_2_P.reset_index(drop=True)
fetch_3_P = fetch_3_P.reset_index(drop=True)
fetch_4_P = fetch_4_P.reset_index(drop=True)

pepper_1_R=exclude(pepper_1_R, R_sublabels)
pepper_2_R=exclude(pepper_2_R, R_sublabels)
pepper_3_R=exclude(pepper_3_R, R_sublabels)
pepper_4_R=exclude(pepper_4_R, R_sublabels)

fetch_1_R=exclude(fetch_1_R, R_sublabels)
fetch_2_R=exclude(fetch_2_R, R_sublabels)
fetch_3_R=exclude(fetch_3_R, R_sublabels)
fetch_4_R=exclude(fetch_4_R, R_sublabels)

pepper_1_P=exclude(pepper_1_P, P_sublabels)
pepper_2_P=exclude(pepper_2_P, P_sublabels)
pepper_3_P=exclude(pepper_3_P, P_sublabels)
pepper_4_P=exclude(pepper_4_P, P_sublabels)

fetch_1_P=exclude(fetch_1_P, P_sublabels)
fetch_2_P=exclude(fetch_2_P, P_sublabels)
fetch_3_P=exclude(fetch_3_P, P_sublabels)
fetch_4_P=exclude(fetch_4_P, P_sublabels) 

#R
def R_mainlabel(df):
    #R
    df['Mainlabel'] = df.loc[:, 'Label']
    df['Mainlabel']=df['Mainlabel'].replace(['introduce background','prepare demonstration'], 'introduction')

    df['Mainlabel']=df['Mainlabel'].replace(['provide clarification','additional info',
                                     'explain behavior','explain resources',
                                     'refer to resources','refer to simulation'], 'clarification')

    df['Mainlabel']=df['Mainlabel'].replace(['prompt action','prompt clarification','prompt evaluation',
                                     'confirm intention','provide opinion',
                                     'summarize discussion','time management','encouragement',
                                     'prompt resources clarification'], 'workshop management')

    df['Mainlabel']=df['Mainlabel'].replace(['propose action','propose approximation',
                                     'propose choice','implement behavior'], 'implementation')

    df['Mainlabel']=df['Mainlabel'].replace(['identify failure','explain failure',
                                     'debugging','identify limitation',
                                     'robot limitation','resources - setup limitation'], 'failure')
#P
def P_mainlabel(df):

    df['Mainlabel'] = df.loc[:, 'Label']
    df['Mainlabel']=df['Mainlabel'].replace(['ask for clarification','accept suggestion',
                                     'accept clarification','call for discussion'], 'information')

    df['Mainlabel']=df['Mainlabel'].replace(['propose role','propose behavior',
                                     'choose behavior','explain proposed behavior',
                                     'clarification reasoning','refer to experience', 'propose action'], 'design action')

    df['Mainlabel']=df['Mainlabel'].replace(['propose replacement','propose fixes',
                                     'propose addition','propose removal',
                                     'identify failure', 'identify limitation'], 'failure action')


    df['Mainlabel']=df['Mainlabel'].replace(['social context','spatial context','user context','liability concern',
                                     'safety concern','ethical concern',
                                     'robot limitation', 'resources - setup limitation'], 'failure reasoning')


    df['Mainlabel']=df['Mainlabel'].replace(['positive','indifferent','anthropomorphize','unsuitable goal',
                                     'interaction - engagement failure','performance failure',
                                     'inappropriate behavior','unexpected behavior', 'refer to simulation'],'perception')
  

R_mainlabel(pepper_1_R)
R_mainlabel(pepper_2_R)
R_mainlabel(pepper_3_R)
R_mainlabel(pepper_4_R)

R_mainlabel(fetch_1_R)
R_mainlabel(fetch_2_R)
R_mainlabel(fetch_3_R)
R_mainlabel(fetch_4_R)

P_mainlabel(pepper_1_P)
P_mainlabel(pepper_2_P)
P_mainlabel(pepper_3_P)
P_mainlabel(pepper_4_P)

P_mainlabel(fetch_1_P)
P_mainlabel(fetch_2_P)
P_mainlabel(fetch_3_P)
P_mainlabel(fetch_4_P) 


# ------- drop & rename columns--------------
def drop_re_col(df):

    return df.drop(['comment','combine'], axis=1).rename(columns={"Label":"Sublabel"})
    

pepper_1_R=drop_re_col(pepper_1_R)
pepper_2_R=drop_re_col(pepper_2_R)
pepper_3_R=drop_re_col(pepper_3_R)
pepper_4_R=drop_re_col(pepper_4_R)

fetch_1_R=drop_re_col(fetch_1_R)
fetch_2_R=drop_re_col(fetch_2_R)
fetch_3_R=drop_re_col(fetch_3_R)
fetch_4_R=drop_re_col(fetch_4_R)

pepper_1_P=drop_re_col(pepper_1_P)
pepper_2_P=drop_re_col(pepper_2_P)
pepper_3_P=drop_re_col(pepper_3_P)
pepper_4_P=drop_re_col(pepper_4_P)

fetch_1_P=drop_re_col(fetch_1_P)
fetch_2_P=drop_re_col(fetch_2_P)
fetch_3_P=drop_re_col(fetch_3_P)
fetch_4_P=drop_re_col(fetch_4_P) 

# ------- index -------------------

pepper_1_P['subindex']=pepper_1_P['Sublabel'].apply(P_sublabels.index)
pepper_2_P['subindex']=pepper_2_P['Sublabel'].apply(P_sublabels.index)
pepper_3_P['subindex']=pepper_3_P['Sublabel'].apply(P_sublabels.index)
pepper_4_P['subindex']=pepper_4_P['Sublabel'].apply(P_sublabels.index)
fetch_1_P['subindex']=fetch_1_P['Sublabel'].apply(P_sublabels.index)
fetch_2_P['subindex']=fetch_2_P['Sublabel'].apply(P_sublabels.index)
fetch_3_P['subindex']=fetch_3_P['Sublabel'].apply(P_sublabels.index)
fetch_4_P['subindex']=fetch_4_P['Sublabel'].apply(P_sublabels.index)

pepper_1_R['subindex']=pepper_1_R['Sublabel'].apply(R_sublabels.index)
pepper_2_R['subindex']=pepper_2_R['Sublabel'].apply(R_sublabels.index)
pepper_3_R['subindex']=pepper_3_R['Sublabel'].apply(R_sublabels.index)
pepper_4_R['subindex']=pepper_4_R['Sublabel'].apply(R_sublabels.index)
fetch_1_R['subindex']=fetch_1_R['Sublabel'].apply(R_sublabels.index)
fetch_2_R['subindex']=fetch_2_R['Sublabel'].apply(R_sublabels.index)
fetch_3_R['subindex']=fetch_3_R['Sublabel'].apply(R_sublabels.index)
fetch_4_R['subindex']=fetch_4_R['Sublabel'].apply(R_sublabels.index)

P_labels=['information','design action', 'failure action','failure reasoning', 'perception']
R_labels=['introduction', 'clarification','workshop management', 'implementation', 'failure']
 
pepper_1_P['index']=pepper_1_P['Mainlabel'].apply(P_labels.index)
pepper_2_P['index']=pepper_2_P['Mainlabel'].apply(P_labels.index)
pepper_3_P['index']=pepper_3_P['Mainlabel'].apply(P_labels.index)
pepper_4_P['index']=pepper_4_P['Mainlabel'].apply(P_labels.index)
fetch_1_P['index']=fetch_1_P['Mainlabel'].apply(P_labels.index)
fetch_2_P['index']=fetch_2_P['Mainlabel'].apply(P_labels.index)
fetch_3_P['index']=fetch_3_P['Mainlabel'].apply(P_labels.index)
fetch_4_P['index']=fetch_4_P['Mainlabel'].apply(P_labels.index)

pepper_1_R['index']=pepper_1_R['Mainlabel'].apply(R_labels.index)
pepper_2_R['index']=pepper_2_R['Mainlabel'].apply(R_labels.index)
pepper_3_R['index']=pepper_3_R['Mainlabel'].apply(R_labels.index)
pepper_4_R['index']=pepper_4_R['Mainlabel'].apply(R_labels.index)
fetch_1_R['index']=fetch_1_R['Mainlabel'].apply(R_labels.index)
fetch_2_R['index']=fetch_2_R['Mainlabel'].apply(R_labels.index)
fetch_3_R['index']=fetch_3_R['Mainlabel'].apply(R_labels.index)
fetch_4_R['index']=fetch_4_R['Mainlabel'].apply(R_labels.index)

# ---------- frames -----------------
ppRframes=[pepper_1_R, pepper_2_R, pepper_3_R, pepper_4_R]

ppPframes=[pepper_1_P, pepper_2_P, pepper_3_P, pepper_4_P]


ftRframes=[fetch_1_R, fetch_2_R, fetch_3_R, fetch_4_R]

ftPframes=[fetch_1_P, fetch_2_P, fetch_3_P, fetch_4_P]

pepper_R=pd.concat(ppRframes)
pepper_P=pd.concat(ppPframes)

fetch_R=pd.concat(ftRframes)
fetch_P=pd.concat(ftPframes)

allRframes=[pepper_1_R, pepper_2_R, pepper_3_R, pepper_4_R, fetch_1_R, fetch_2_R, fetch_3_R, fetch_4_R]
allPframes=[pepper_1_P, pepper_2_P, pepper_3_P, pepper_4_P, fetch_1_P, fetch_2_P, fetch_3_P, fetch_4_P]

all_R=pd.concat(allRframes)
all_P=pd.concat(allPframes)

pepper_1_merge=pd.concat([pepper_1_R, pepper_1_P])
pepper_2_merge=pd.concat([pepper_2_R, pepper_2_P])
pepper_3_merge=pd.concat([pepper_3_R, pepper_3_P])
pepper_4_merge=pd.concat([pepper_4_R, pepper_4_P])

fetch_1_merge=pd.concat([fetch_1_R, fetch_1_P])
fetch_2_merge=pd.concat([fetch_2_R, fetch_2_P])
fetch_3_merge=pd.concat([fetch_3_R, fetch_3_P])
fetch_4_merge=pd.concat([fetch_4_R, fetch_4_P])

# ------------ reset index ------------------
pepper_1_merge = pepper_1_merge.reset_index(drop=True)
pepper_2_merge = pepper_2_merge.reset_index(drop=True)
pepper_3_merge = pepper_3_merge.reset_index(drop=True)
pepper_4_merge = pepper_4_merge.reset_index(drop=True)

fetch_1_merge = fetch_1_merge.reset_index(drop=True)
fetch_2_merge = fetch_2_merge.reset_index(drop=True)
fetch_3_merge = fetch_3_merge.reset_index(drop=True)
fetch_4_merge = fetch_4_merge.reset_index(drop=True)

pepper_R = pepper_R.reset_index(drop=True)
pepper_P = pepper_P.reset_index(drop=True)
fetch_R = fetch_R.reset_index(drop=True)
fetch_P = fetch_P.reset_index(drop=True)

all_R = all_R.reset_index(drop=True)
all_P = all_P.reset_index(drop=True)


pepper_1_R = pepper_1_R.reset_index(drop=True)
pepper_2_R = pepper_2_R.reset_index(drop=True)
pepper_3_R = pepper_3_R.reset_index(drop=True)
pepper_4_R = pepper_4_R.reset_index(drop=True)

fetch_1_R = fetch_1_R.reset_index(drop=True)
fetch_2_R = fetch_2_R.reset_index(drop=True)
fetch_3_R = fetch_3_R.reset_index(drop=True)
fetch_4_R = fetch_4_R.reset_index(drop=True)

pepper_1_P = pepper_1_P.reset_index(drop=True)
pepper_2_P = pepper_2_P.reset_index(drop=True)
pepper_3_P = pepper_3_P.reset_index(drop=True)
pepper_4_P = pepper_4_P.reset_index(drop=True)

fetch_1_P = fetch_1_P.reset_index(drop=True)
fetch_2_P = fetch_2_P.reset_index(drop=True)
fetch_3_P = fetch_3_P.reset_index(drop=True)
fetch_4_P = fetch_4_P.reset_index(drop=True)

# -------- save files --------------
pepper_1_merge.to_csv('./data/processed/pepper_1_merge.csv')
pepper_2_merge.to_csv('./data/processed/pepper_2_merge.csv')
pepper_3_merge.to_csv('./data/processed/pepper_3_merge.csv')
pepper_4_merge.to_csv('./data/processed/pepper_4_merge.csv')

fetch_1_merge.to_csv('./data/processed/fetch_1_merge.csv')
fetch_2_merge.to_csv('./data/processed/fetch_2_merge.csv')
fetch_3_merge.to_csv('./data/processed/fetch_3_merge.csv')
fetch_4_merge.to_csv('./data/processed/fetch_4_merge.csv')

pepper_R.to_csv('./data/processed/pepper_R.csv')
pepper_P.to_csv('./data/processed/pepper_P.csv')
fetch_R.to_csv('./data/processed/fetch_R.csv')
fetch_P.to_csv('./data/processed/fetch_P.csv')

all_R.to_csv('./data/processed/all_R.csv')
all_P.to_csv('./data/processed/all_P.csv')

pepper_1_R.to_csv('./data/processed/pepper_1_R.csv')
pepper_2_R.to_csv('./data/processed/pepper_2_R.csv')
pepper_3_R.to_csv('./data/processed/pepper_3_R.csv')
pepper_4_R.to_csv('./data/processed/pepper_4_R.csv')

fetch_1_R.to_csv('./data/processed/fetch_1_R.csv')
fetch_2_R.to_csv('./data/processed/fetch_2_R.csv')
fetch_3_R.to_csv('./data/processed/fetch_3_R.csv')
fetch_4_R.to_csv('./data/processed/fetch_4_R.csv')

pepper_1_P.to_csv('./data/processed/pepper_1_P.csv')
pepper_2_P.to_csv('./data/processed/pepper_2_P.csv')
pepper_3_P.to_csv('./data/processed/pepper_3_P.csv')
pepper_4_P.to_csv('./data/processed/pepper_4_P.csv')

fetch_1_P.to_csv('./data/processed/fetch_1_P.csv')
fetch_2_P.to_csv('./data/processed/fetch_2_P.csv')
fetch_3_P.to_csv('./data/processed/fetch_3_P.csv')
fetch_4_P.to_csv('./data/processed/fetch_4_P.csv')


