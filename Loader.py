from transformers import AutoTokenizer
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
 
class TextDataLoader(Dataset):
    def __init__(self, text, labels, tokenizer):
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        
        text = self.text[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )


        return {
            'input' : encoding['input_ids'].squeeze(0),
            'attn_mask' : encoding['attention_mask'].squeeze(0),
            'label' : torch.tensor(label, dtype = torch.long),
        }