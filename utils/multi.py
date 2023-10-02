'''
functions for multimodal model
1. dataset, dataloader
2. train model
3. get predictions
'''

from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import cv2
from setting import config

class MultiDataSet(Dataset):
    def __init__(self, img_names, img_dir, text, target, tokenizer, transforms):
      super(MultiDataSet, self).__init__()
      self.img_names = img_names+'.jpg'
      self.img_dir = img_dir
      self.text = list(text)
      self.target = target
      self.tokenizer= tokenizer
      self.transforms = transforms
      self.encoded_captions = tokenizer(
            self.text, padding=True, truncation=True, max_length=config.max_length
        )

    def __getitem__(self, index):
      item = { key: torch.tensor(values[index])
            for key, values in self.encoded_captions.items()} 
      image = cv2.imread(f"{self.img_dir}{self.img_names[index]}")
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = self.transforms(image=image)['image']
      item['target'] = self.target[index]
      item['image'] = torch.tensor(image).permute(2, 0, 1).float()

      return item

    def __len__(self):
      return len(self.text)
    

def create_multi_loader(df, img_dir, shuffle=True):
    ds = MultiDataSet(text=df['text'].values,
                      target=df['index'].values,
                      img_names= df['file_name'].values,
                      img_dir=img_dir,
                      tokenizer=config.tokenizer,
                      transforms=config.transforms)

    return DataLoader(ds,
                      batch_size=config.batch_size
                      ,shuffle=shuffle)
    

def train_epoch(model, trainloader, loss_fn, optimizer, device, scheduler, n_examples):
    model=model.train()
    model = model.to(device)

    losses=[]
    correct_predictions = 0

    for batch in tqdm(trainloader):
        optimizer.zero_grad()
        label=batch['label'].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audio=batch['image'].to(device)

        output = model(input_ids,attention_mask,audio)
        _, preds = torch.max(output, dim=1)
        loss = loss_fn(output, label)

        correct_predictions += torch.sum(preds == label)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, device, testloader):
    model=model.eval()
    model=model.to(device)

    predictions = []
    real_values = []

    with torch.no_grad():
        for d in testloader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            label = d["label"].to(device)
            audio=d['image'].to(device)

            outputs = model(
                input_ids,attention_mask,audio
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)


            predictions.extend(preds)

            real_values.extend(label)

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return predictions, real_values