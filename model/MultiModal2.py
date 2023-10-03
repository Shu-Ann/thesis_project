'''
text & image models from pretrained BERT & resnet152 
'''
import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import BertModel
from torchvision.models import resnet152

class Bert(nn.Module):
    
  def __init__(self):
    super(Bert, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased')
  

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


class AudioTextEncodeModel(nn.Module):
    def __init__(self, num_classes):
        super(AudioTextEncodeModel, self).__init__()
        self.num_classes=num_classes
        self.text_model=Bert()
        self.audio_premodel=Resnet152()

        self.dropout = nn.Dropout(.5)
        self.fc1 = nn.Linear(2816,num_classes)

    def forward(self,input_ids,attention_mask, audio):
        outputs_text=self.text_model(input_ids, attention_mask)
        outputs_audio=self.audio_premodel(audio)
        outputs_audio=outputs_audio.flatten(1)
        concat_embded=torch.cat((outputs_text,outputs_audio),1)
        preds = self.fc1(self.dropout(concat_embded))
        return preds