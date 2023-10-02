'''
split train/ valid/ test data
'''

import pandas as pd
from sklearn.model_selection import train_test_split

df_p = pd.read_csv("./data/model_use/all_P.csv")
df_r = pd.read_csv("./data/model_use/all_R.csv")

df_p=df_p.rename(columns={'Mainlabel':'label'})
df_r=df_r.rename(columns={'Mainlabel':'label'})

# ---- train/test ------- 80/20
train_p, test_p = train_test_split(df_p, test_size=0.2)
train_r, test_r = train_test_split(df_r, test_size=0.2)
# --- valid/test ---- 50/50
test_p, valid_p = train_test_split(test_p, test_size=0.5)
test_r, valid_r = train_test_split(test_r, test_size=0.5)

test_r = test_r.reset_index(drop=True)
train_r = train_r.reset_index(drop=True)
test_p = test_p.reset_index(drop=True)
train_p = train_p.reset_index(drop=True)
valid_p = valid_p.reset_index(drop=True)
valid_r = valid_r.reset_index(drop=True)


valid_p.to_csv('./data/model_use/valid_p.csv')
valid_r.to_csv('./data/model_use/valid_r.csv') 

test_p.to_csv('./data/model_use/test_p.csv')
train_p.to_csv('./data/model_use/train_p.csv') 

test_r.to_csv('./data/model_use/test_r.csv')
train_r.to_csv('./data/model_use/train_r.csv') 