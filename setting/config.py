import torch
from transformers import BertTokenizer, BertModel
import albumentations as A
from torch import nn, optim
import pandas as pd

# ---------  MODE / ROLE -------------
mode='text'
role='p'
# ---------  MODE / ROLE -------------

# data
train_r = pd.read_csv('../data/model_use/train_r.csv')
train_p = pd.read_csv('../data/model_use/train_p.csv')

valid_r = pd.read_csv('../data/model_use/valid_r.csv')
valid_p = pd.read_csv('../data/model_use/valid_p.csv')

test_r = pd.read_csv('../data/model_use/test_r.csv')
test_p = pd.read_csv('../data/model_use/test_p.csv')

# img_dir path
train_img_r = '../data/model_use/train_R_image/'
train_img_p = '../data/model_use/train_P_image/'

test_img_r = '../data/model_use/test_R_image/'
test_img_p = '../data/model_use/test_P_image/'

# labels
P_labels=['information','design action', 'failure action','failure reasoning', 'perception']
R_labels=['introduction', 'clarification','workshop management', 'implementation', 'failure']

#device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#batch size
batch_size = 16

# loss function
loss_fn = nn.CrossEntropyLoss().to(device)
# -------------------- text --------------------
# tokenizer/ bert model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
text_pretrained = BertModel.from_pretrained('bert-base-cased')

# params
bert_lr = 2e-5
max_length=256
epoch = 20
patience = 2
factor = 0.5


# -------------------- ResNet--------------------

# transformers
transformers = transforms=A.Compose(
            [A.Resize(255, 255, always_apply=True),
             A.Normalize(max_pixel_value=255.0, always_apply=True)])

res_lr = 0.001
res152_factor=0.3


# -------------------- Multi--------------------

# -------------------- path --------------------

R_text_model_path='./model/R_text_model.pt'
R_audio_model_path='./model/R_audio_model.pt'

P_text_model_path='./model/P_text_model.pt'
P_audio_model_path='./model/P_audio_model.pt'


