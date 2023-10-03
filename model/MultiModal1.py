'''
trained 80-20 text model + trained 80-20 CNN model
'''
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision.models import resnet50

class Bert(nn.Module):
    
  def __init__(self, text_model_path):
    super(Bert, self).__init__()
    self.bert = torch.load(text_model_path)
    self.bert.load_state_dict(self.bert.state_dict())

  def forward(self, input_ids, attention_mask):
    output=self.bert(input_ids=input_ids,attention_mask=attention_mask)
    return output

class Resnet152(nn.Module):
    
  def __init__(self,audio_model_path):
    super(Resnet152, self).__init__()
    self.model = torch.load(audio_model_path)

  def forward(self, image):
    out = self.model(image)
    return out

class Resnet50(nn.Module):
    
  def __init__(self,audio_model_path):
    super(Resnet50, self).__init__()
    self.model = torch.load(audio_model_path)

  def forward(self, image):
    out = self.model(image)

    return out


class AudioTextModel(nn.Module):
    def __init__(self, num_classes, text_model_path, audio_model_path):
        super(AudioTextModel, self).__init__()
        self.num_classes=num_classes

        self.text_model=torch.load(text_model_path)
        self.audio_premodel=torch.load(audio_model_path)
        modules=list(self.audio_premodel.children())[:-1]
        self.audio_model=nn.Sequential(*modules)


        for param in self.text_model.parameters():
          param.requires_grad = False
        for param in self.audio_premodel.parameters():
          param.requires_grad = False

        self.dropout = nn.Dropout(.5)
        self.fc = nn.Linear(2816,num_classes)

    def forward(self,input_ids,attention_mask, audio):
        output=self.text_model(input_ids, attention_mask)
        last_hidden_state = output.last_hidden_state
        outputs_text=last_hidden_state[:, 0, :]

        outputs_audio=self.audio_model(audio)
        outputs_audio=outputs_audio.flatten(1)

        concat_embded=torch.cat((outputs_text,outputs_audio),1)
        preds = self.fc(self.dropout(concat_embded))

        return preds
