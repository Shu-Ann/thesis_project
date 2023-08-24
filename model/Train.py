from Bert import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
import pandas as pd

from torch import nn, optim

os.environ['CURL_CA_BUNDLE'] = ''
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

R_mainlabels=['introduction', 'clarification','workshop management', 'implementation', 'failure']
P_mainlabels=['information','design action', 'failure action','failure reasoning', 'perception']

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


# ------  helper functions  ---------
# --dataloader--
# mainlabel
def mainlabel_create_data_loader(df, tokenizer, max_len, batch_size):
      ds = MainDataset(text=df['text'].to_numpy(),
                       targets=df['index'].to_numpy(),
                       tokenizer=tokenizer,
                       max_len=max_len)
      return DataLoader(ds, batch_size=batch_size)
#sub-label
def sublabel_create_data_loader(df, tokenizer, max_len, batch_size):
      ds = MainDataset(text=df['text'].to_numpy(),
                       targets=df['subindex'].to_numpy(),
                       tokenizer=tokenizer,
                       max_len=max_len)
      return DataLoader(ds, batch_size=batch_size)

# --heatmap--
def show_confusion_matrix(confusion_matrix):
     hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
     hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
     hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
     plt.ylabel('True')
     plt.xlabel('Predicted')

# -- run epoch -----
def training_process(epoch, Model):
    for epoch in range(EPOCHS):
    
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

    train_acc, train_loss = Model.train_epoch(train_data_loader_r)

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = Model.eval_model(val_data_loader_r)

    print(f'Val loss {val_loss} accuracy {val_acc}')
    print()

# define parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
loss_fn = nn.CrossEntropyLoss().to(device)

df_r=pd.read_csv('./data/processed/all_R.csv')
df_p=pd.read_csv('./data/processed/all_P.csv')

# Researcher 

train_r, test_r = train_test_split(df_r, test_size=0.2)
valid_r, test_r = train_test_split(test_r, test_size=0.5)

train_data_loader_r = mainlabel_create_data_loader(train_r, tokenizer, max_len=128, batch_size=16)
val_data_loader_r = mainlabel_create_data_loader(valid_r, tokenizer, max_len=128, batch_size=16)
test_data_loader_r = mainlabel_create_data_loader(test_r, tokenizer, max_len=128, batch_size=16)

model = Classifier(len(R_mainlabels))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader_r) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

R_model_train=Train_Model(model, loss_fn, optimizer, device, scheduler, len(train_r))

training_process(EPOCHS, R_model_train)

# Participant 

train_p, test_p = train_test_split(df_p, test_size=0.2)
valid_p, test_p = train_test_split(test_p, test_size=0.5)

train_data_loader_p = mainlabel_create_data_loader(train_p, tokenizer, max_len=128, batch_size=16)
val_data_loader_p = mainlabel_create_data_loader(valid_p, tokenizer, max_len=128, batch_size=16)
test_data_loader_p = mainlabel_create_data_loader(test_p, tokenizer, max_len=128, batch_size=16)

model = Classifier(len(P_mainlabels))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader_p) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

P_model_train=Train_Model(model, loss_fn, optimizer, device, scheduler, len(train_p))

training_process(EPOCHS, P_model_train)