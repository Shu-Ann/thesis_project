'''
functions for bert model
1. dataset, dataloader
2. train, eval model
3. get predictions
'''
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from setting import config

class TextDataset(Dataset):
    
    def __init__(self, text, targets, tokenizer):
        
        self.targets = targets
        self.text = list(text)
        self.tokenizer = tokenizer
        self.encoded_captions = tokenizer(self.text, padding=True, truncation=True, max_length=config.max_length)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(values[index])
            for key, values in self.encoded_captions.items()
        } 
  
        item['target'] = self.targets[index]

        return item
    

def create_text_loader(df, shuffle=True):
    ds = TextDataset(text=df['text'].values,
                     targets=df['index'].values,
                     tokenizer=config.tokenizer)

    return DataLoader(ds,
                      batch_size=config.batch_size
                      ,shuffle=shuffle)


def train_epoch(model, train_dataloader, optimizer, scheduler, n_train):
    model = model.train()
    losses = []
    correct_predictions = 0
    process_bar = tqdm(train_dataloader)
    for d in process_bar:
        d = {k: v.to(config.device) for k, v in d.items() if k != "text"}
        input_ids = d["input_ids"].to(config.device)
        attention_mask = d["attention_mask"].to(config.device)
        targets = d['target'].to(config.device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = config.loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        process_bar.set_postfix(train_loss=loss.item())

    return correct_predictions.double() / n_train, np.mean(losses)

def eval_model(model, val_dataloader, n_valid):
    model = model.eval()

    losses = []
    correct_predictions = 0
    process_bar = tqdm(val_dataloader)
    for d in process_bar:
        d = {k: v.to(config.device) for k, v in d.items() if k != "text"}
        input_ids = d["input_ids"].to(config.device)
        attention_mask = d["attention_mask"].to(config.device)
        targets = d["target"].to(config.device)

        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

        loss = config.loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        process_bar.set_postfix(val_loss=loss.item())

    return correct_predictions.double() / n_valid, np.mean(losses)


def get_predictions(model, test_data_loader):
    model = model.eval()
    predictions = []
    real_values = []

    for d in test_data_loader:
        d = {k: v.to(config.device) for k, v in d.items() if k != "text"}
        input_ids = d["input_ids"].to(config.device)
        attention_mask = d["attention_mask"].to(config.device)
        targets = d["target"].to(config.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)

        probs = F.softmax(outputs, dim=1)

        predictions.extend(preds)
        real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return predictions, real_values



