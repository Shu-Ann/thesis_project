import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from transformers import *


## colab
# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8
model="typeform/distilbert-base-uncased-mnli"

def show_confusion_matrix(confusion_matrix):
   hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
   hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
   hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
   plt.ylabel('True')
   plt.xlabel('Predicted')

def model_pipeline(df, candidate_labels):

  classifier=pipeline("zero-shot-classification",model=model)

  predict=[]
  true=[]
  for i in tqdm(range(0, len(df))):
    text = df.iloc[i,]['text']
    cat = df.iloc[i,]['label']
    res = classifier(text, candidate_labels, multi_label=False)
    labels = res['labels'][0]

    predict.append(labels)
    true.append(cat)
    
  cm = confusion_matrix(true, predict)
  df_cm = pd.DataFrame(cm, index=candidate_labels, columns=candidate_labels)
  show_confusion_matrix(df_cm)

  return(classification_report(true, predict, target_names=candidate_labels))