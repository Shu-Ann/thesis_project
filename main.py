'''
train and save the model pipeline
'''
from setting import config
from utils.audio import create_audio_loader
from utils.berthelper import create_text_loader
from utils.multi import create_multi_loader
from utils import train
from model import *



if config.mode == "text":
    train_dataloader = create_text_loader(ds, shuffle=True)
    eval_dataloader = create_text_loader(ds, shuffle=True)
    test_dataloader = create_text_loader(ds, shuffle=False)

elif config.mode == "audio":
    train_dataloader = create_audio_loader(ds, shuffle=True)
    eval_dataloader = create_audio_loader(ds, shuffle=True)
    test_dataloader = create_audio_loader(ds, shuffle=False)

elif config.mode == "multi":
    train_dataloader = create_multi_loader(ds, shuffle=True)
    eval_dataloader = create_multi_loader(ds, shuffle=True)
    test_dataloader = create_multi_loader(ds, shuffle=False)

 
# ------   text 5/10 shots & 80-20  -------
# train model -> save model

# ------ audio resnet50/ resnet152---------
# train model -> save model 

# ---------- multimodal ------------
# concat text embedding & audio embedding -> train model -> save model 



