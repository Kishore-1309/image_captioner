import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
from tqdm import tqdm
from .dataset import CaptionDataset
from .model import TransformerBridge, CLIPGPT2CaptionModel
from .utils import save_checkpoint
from .config import Config

def train(feature_dir, caption_csv, batch_size=16, num_epochs=20, lr=5e-5):
    config = Config()
    device = config.DEVICE
    
    # Initialize components
    df = pd.read_csv(caption_csv)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = CaptionDataset(df, tokenizer, feature_dir, config.MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model setup
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    bridge = TransformerBridge()
    model = CLIPGPT2CaptionModel(bridge, gpt2).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # Save checkpoint
        avg_loss = total_loss / len(dataloader)
        checkpoint_path = save_checkpoint(model, optimizer, epoch+1, config.DEFAULT_CHECKPOINT_DIR)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Saved to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--captions", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    
    train(args.features, args.captions, args.batch_size, args.epochs, args.lr)