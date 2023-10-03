'''
training step only for text/audio Models
'''

from setting import config
import torch
from model.BertModel import Bert
from model.AudioCNNModel import Resnet152, Resnet50
from transformers import AdamW
from utils import berthelper, audio
from utils.common import *

def single_train(config,train_dataloader, val_dataloader,test_dataloader, labels, n_train, n_valid, path):
    
    if config.mode=='text':
        model=Bert(5).to(config.device)
        optimizer = AdamW(model.parameters(), lr=config.bert_lr,correct_bias=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.patience, factor=config.factor)
 
    
        for epoch in range(config.epoch):
        
            print(f'Epoch {epoch + 1}/{config.epoch}')
            print('-' * 10)

            train_acc, train_loss = berthelper.train_epoch(model, 
                                                           train_dataloader, 
                                                           optimizer, 
                                                           n_train 
                                                           )
            print(f'Train loss {train_loss} accuracy {train_acc}')
            with torch.no_grad():
                val_acc, val_loss = berthelper.eval_model(model, val_dataloader, n_valid)

            print(f'Val loss {val_loss} accuracy {val_acc}')
            print()
            scheduler.step(val_loss)

        saveModel(model, path)
        with torch.no_grad():
            y_pred, y_test=berthelper.get_predictions(model, test_dataloader)

    elif config.mode=='audio':
        model=Resnet152(5).to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.res_lr )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.patience, factor=config.res152_factor)

        for e in range(config.epoch):
    
            print(f'Epoch {e + 1}/{config.epoch}')
            print('-' * 10)
            train_acc, train_loss= audio.train_epoch(model, 
                                                     train_dataloader, 
                                                     optimizer, 
                                                     n_train)

            print(f'Train loss {train_loss} accuracy {train_acc}')
            with torch.no_grad():
                val_acc ,val_loss= audio.eval_model(model,val_dataloader, n_valid)
            

            print(f'Val loss {val_loss} accuracy {val_acc}')
            print()
            scheduler.step(val_loss)

        saveModel(model, path)
        with torch.no_grad():
            y_pred, y_test=audio.get_predictions(model, test_dataloader)

    show_confusion_matrix(y_test, y_pred, labels)
    return report(y_test, y_pred, labels)




