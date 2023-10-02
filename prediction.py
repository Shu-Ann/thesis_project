'''
predict and lanuch web
'''

# from torch.utils.data import Dataset, DataLoader
import torch
# from torch import nn
# from tqdm import tqdm
# import numpy as np
# import torch.nn.functional as F
from setting import config


def predict_text(query):
    model = model.eval()
    tokenizer = config.tokenizer
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(config.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        output = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'])
        
        _, preds = torch.max(output, dim=1)