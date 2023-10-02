'''
convert audio files to MFCC image files
'''


import librosa
import librosa.display
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def create_melspectrogram_train(filename,name, mode):
    
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    if mode =='P':
        filename  = ('./data/model_use/train_P_image/' + name + '.jpg')
    elif mode=='R':
        filename  = ('./data/model_use/train_R_image/' + name + '.jpg')
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

def create_melspectrogram_test(filename,name, mode):

    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    if mode =='P':
        filename  = ('./data/model_use/test_P_image/' + name + '.jpg')
    elif mode=='R':
        filename  = ('./data/model_use/test_R_image/' + name + '.jpg')
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


# --- reseacher -----

Data_dir_train=np.array(glob("./data/audio/train_R_audio/*"))
Data_dir_test=np.array(glob("./data/audio/test_R_audio/*"))

for file in tqdm(Data_dir_train):
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_melspectrogram_train(filename,name, 'R')
for file in tqdm(Data_dir_test):
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_melspectrogram_test(filename,name, 'R')

del Data_dir_train, Data_dir_test

# --- participant -----

Data_dir_train=np.array(glob("./data/audio/train_P_audio/*"))
Data_dir_test=np.array(glob("./data/audio/test_P_audio/*"))

for file in tqdm(Data_dir_train):
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_melspectrogram_train(filename,name, 'P')
for file in tqdm(Data_dir_test):
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_melspectrogram_test(filename,name, 'P')

