'''
functions for audio model
1. dataset, dataloader
2. train, eval model
3. get predictions
'''
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import cv2
from setting import config

class AudioCNNDataset(Dataset):
    def __init__(self, img_names, img_dir, target, transforms):
      super(AudioCNNDataset, self).__init__()
      self.img_names = img_names+'.jpg'
      self.img_dir = img_dir
      self.target = target
      self.transforms = transforms

    def __getitem__(self, index):
      item = {}
      image = cv2.imread(f"{self.img_dir}{self.img_names[index]}")
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = self.transforms(image=image)['image']
      item['target'] = self.target[index]
      item['image'] = torch.tensor(image).permute(2, 0, 1).float()

      return item


    def __len__(self):
      return len(self.img_names)
    

def create_audio_loader(df, img_dir, shuffle=True):
  ds = AudioCNNDataset(target=df['index'].values,
                       img_names= df['file_name'].values,
                       img_dir=img_dir,
                       transforms=config.transforms)

  return DataLoader(ds,
                    batch_size=config.batch_size
                    ,shuffle=shuffle)

    
def train_epoch(model, trainloader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train() # puts the model in training mode
    correct=0
    running_loss = 0.0

    for batch in tqdm(trainloader):
      targets = batch["target"].to(device)
      audio=batch['image'].to(device)
      optimizer.zero_grad() # clear the gradients in model parameters
      logits, probas = model(audio) # put data into model to predict
      loss = loss_fn(logits, targets) # calculate loss between prediction and true labels
      loss.backward() # back propagation: pass the loss
      optimizer.step()  # iterate over all parameters in the model with requires_grad=True and update their weights.
      scheduler.step()
      # compute training statistics
      _, predicted = torch.max(logits, 1)
      correct += (predicted == targets).sum().item()
      running_loss += loss.item() # sum total loss in current epoch for print later

    avg_loss = running_loss / n_examples
    avg_acc = correct / n_examples

    return avg_loss, avg_acc

def eval_model(model, validloader, loss_fn, device, n_examples):
    model=model.eval() # puts the model in validation mode
    with torch.no_grad():
        loss_val = 0.0
        correct_val = 0
    for data in tqdm(validloader):
        batch, labels = data
        batch, labels = batch.to(device), labels.to(device)
        logits, probas = model(batch)
        loss = loss_fn(logits, labels)
        _, predicted = torch.max(logits, 1)
        correct_val += (predicted == labels).sum().item()
        loss_val += loss.item()
    avg_loss_val = loss_val / n_examples
    avg_acc_val = correct_val /n_examples

    return avg_loss_val, avg_acc_val

def get_predictions(model, device, testloader):
  model=model.eval()
  model.to(device)
  predictions=[]
  real_values=[]
  with torch.no_grad():
    for batch in tqdm(testloader):
      targets = batch["target"].to(device)
      audio=batch['image'].to(device)

      logits, probas = model(audio)
      _, predicted_labels = torch.max(probas, 1)

      predictions.extend(predicted_labels)
      real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()

    return predictions, real_values

def predict(model, device, audio):
    model=model.eval()
    model.to(device)
    with torch.no_grad():
        logits, probas= model(audio)

    return 
