import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class CaptionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, feature_dir, max_length=40):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.feature_dir = feature_dir
        self.max_length = max_length
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row["image"]
        caption = row["caption"]
        
        if not isinstance(caption, str):
            caption = str(caption) if caption is not None else " "

        # Add BOS and EOS tokens to the caption
        caption = self.tokenizer.bos_token + caption + self.tokenizer.eos_token

        
        # Load image features
        feature_path = os.path.join(self.feature_dir, image_name.replace(".jpg", ".pt"))
        image_features = torch.load(feature_path,weights_only=True)
        
        # Tokenize caption
        encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0),
            "image_features": image_features
        }