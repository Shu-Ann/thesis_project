'''
functions for all models
1. heatmap
2. balanced accuracy
3. classification report
'''

from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import seaborn as sns
from pylab import rcParams
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_inline
import torch

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

def plot_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

def show_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    
    return plot_confusion_matrix(df_cm)

def balanced_accuracy(y_test, y_pred):
    acc=balanced_accuracy_score(y_test, y_pred)
    return acc

def report(y_test, y_pred, labels):
    print(classification_report(y_test, y_pred, target_names=labels))


def saveModel(model, path):
    torch.save(model, path)
