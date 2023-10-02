'''
trained 80-20 text model + trained 80-20 resnet152 model
'''
import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import BertModel
from torchvision.models import resnet152

class Bert(nn.Module):
    
  def __init__(self, n_classes):
    super(Bert, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased')
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    output= self.bert(input_ids=input_ids,attention_mask=attention_mask)
    last_hidden_state = output.last_hidden_state
    return last_hidden_state[:, 0, :]

class Resnet152(nn.Module):
    
  def __init__(self):
    super(Resnet152, self).__init__()
    self.premodel = resnet152(weights="IMAGENET1K_V2")
    modules=list(self.premodel.children())[:-1]
    self.model=nn.Sequential(*modules)

  def forward(self, image):
    out = self.model(image)

    return out


class AudioTextModel(nn.Module):
    def __init__(self, num_classes, text_model_path, audio_model_path):
        super(AudioTextModel, self).__init__()
        self.num_classes=num_classes

        self.text_model=torch.load(text_model_path)
        self.audio_premodel=torch.load(audio_model_path)

        # for param in self.text_model.parameters():
        #   param.requires_grad = False
        # for param in self.audio_model.parameters():
        #   param.requires_grad = False

        self.dropout = nn.Dropout(.5)
        self.fc1 = nn.Linear(2816,1200)
        self.fc2 = nn.Linear(1200,600)
        self.fc3 = nn.Linear(600,300)
        self.fc4 = nn.Linear(300,num_classes)

    def forward(self,input_ids,attention_mask, audio):
        outputs_text=self.text_model(input_ids, attention_mask)
        outputs_audio=self.audio_premodel(audio)
        outputs_audio=outputs_audio.flatten(1)
        concat_embded=torch.cat((outputs_text,outputs_audio),1)
        l1 = self.fc1(self.dropout(concat_embded))
        l2 = self.fc2(l1)
        l3 = self.fc3(l2)
        preds = self.fc4(l3)
        return preds