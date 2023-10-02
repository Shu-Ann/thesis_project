'''
text augmentation 
using terminal
'''
import pandas as pd
from sklearn.utils import shuffle
train_p = pd.read_csv("./data/model_use/train_p.csv")
train_r = pd.read_csv("./data/model_use/train_r.csv")
df_r_4=train_r[(train_r['label']=='introduction')| (train_r['label']=='failure')].to_csv('./data/text_aug/R_4.csv')
df_r_2=train_r[(train_r['label']=='implementation')].to_csv('./data/text_aug/R_2.csv')
df_r_1=train_r[(train_r['label']=='clarification')].to_csv('./data/text_aug/R_1.csv')


df_p_3=train_p[(train_p['label']=='failure action') |(train_p['label']=='information')].to_csv('./data/text_aug/P_3.csv')
df_p_1=train_p[(train_p['label']=='perception')].to_csv('./data/text_aug/P_1.csv')

# ------- terminal ---------
#!pip install textattack
#!textattack augment --input-csv ./data/text_aug/R_4.csv --output-csv ./data/text_aug/R_4_attack.csv  --input-column text --recipe embedding --pct-words-to-swap .9 --transformations-per-example 4 --exclude-original
#!textattack augment --input-csv ./data/text_aug/R_1.csv --output-csv ./data/text_aug/R_1_attack.csv  --input-column text --recipe embedding --pct-words-to-swap .9 --transformations-per-example 1 --exclude-original
#!textattack augment --input-csv ./data/text_aug/P_3.csv --output-csv ./data/text_aug/P_3_attack.csv  --input-column text --recipe embedding --pct-words-to-swap .9 --transformations-per-example 3 --exclude-original
#!textattack augment --input-csv ./data/text_aug/P_1.csv --output-csv ./data/text_aug/P_1_attack.csv  --input-column text --recipe embedding --pct-words-to-swap .9 --transformations-per-example 1 --exclude-original
P_labels=['information','design action', 'failure action','failure reasoning', 'perception']
R_labels=['introduction', 'clarification','workshop management', 'implementation', 'failure']

R_1_attack=pd.read_csv('./data/text_aug/R_1_attack.csv')
R_2_attack=pd.read_csv('./data/text_aug/R_2_attack.csv')
R_4_attack=pd.read_csv('./data/text_aug/R_4_attack.csv')

P_3_attack=pd.read_csv("./data/text_aug/P_3_attack.csv")
P_1_attack=pd.read_csv("./data/text_aug/P_1_attack.csv")

failure=R_4_attack[(R_4_attack['label']=='failure')] #606
introduction=R_4_attack[(R_4_attack['label']=='introduction')]
implementation=R_2_attack[(R_2_attack['label']=='implementation')] # 493
clarification=R_1_attack[(R_1_attack['label']=='clarification')] #282

train_r=pd.concat([failure[:606] , introduction, implementation[:493], clarification[:282],train_r])

failure_action=P_3_attack[(P_3_attack['label']=='failure action')] #330
perception=P_1_attack[(P_1_attack['label']=='perception')] #152
information=P_3_attack[(P_3_attack['label']=='information')] #250
train_p=pd.concat([failure_action[:330] , perception[:152], information[:250],train_p])

train_p_aug = shuffle(train_p).reset_index(drop=True)
train_r_aug = shuffle(train_r).reset_index(drop=True)

train_p_aug.to_csv('./data/model_use/train_p_aug.csv')
train_r_aug.to_csv('./data/model_use/train_r_aug.csv')


