'''
pre-trained resnet-50 & resnet-152
'''
from torchvision.models import resnet50, resnet152
from torch import nn
import torch.nn.functional as F

class Resnet50(nn.Module):
    
  def __init__(self, n_classes):
    super(Resnet50, self).__init__()
    self.premodel = resnet50(weights="IMAGENET1K_V2")
    modules=list(self.premodel.children())[:-1]
    self.model=nn.Sequential(*modules)
    self.fc=nn.Linear(2048, n_classes)

  def forward(self, image):
    out = self.model(image)
    output=out.flatten(1)
    logits=self.fc(output)
    probas = F.softmax(logits, dim=1)

    return logits, probas
  
class Resnet152(nn.Module):
    
  def __init__(self, n_classes):
    super(Resnet152, self).__init__()
    self.premodel = resnet152(weights="IMAGENET1K_V2")
    modules=list(self.premodel.children())[:-1]
    self.model=nn.Sequential(*modules)
    self.fc=nn.Linear(2048, n_classes)

  def forward(self, image):
    out = self.model(image)
    output=out.flatten(1)
    logits=self.fc(output)
    probas = F.softmax(logits, dim=1)

    return logits, probas