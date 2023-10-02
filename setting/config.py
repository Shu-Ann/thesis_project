import torch
from transformers import BertTokenizer, BertModel
import albumentations as A
from torch import nn, optim

# ---------  MODE -------------
mode='text'
# ---------  MODE -------------



# labels
P_labels=['information','design action', 'failure action','failure reasoning', 'perception']
R_labels=['introduction', 'clarification','workshop management', 'implementation', 'failure']

#device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#batch size
batch_size = 16
# -------------------- text --------------------
# tokenizer/ bert model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
text_pretrained = BertModel.from_pretrained('bert-base-cased')

# params
bert_lr = 2e-5
max_length=256
epoch = 20
patience= 2
factor=0.5
# loss function
loss_fn = nn.CrossEntropyLoss().to(device)

# -------------------- ResNet--------------------

res_lr = 0.001
transformers = transforms=A.Compose(
            [A.Resize(255, 255, always_apply=True),
             A.Normalize(max_pixel_value=255.0, always_apply=True)])



# -------------------- Multi--------------------

# -------------------- path --------------------



