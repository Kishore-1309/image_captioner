import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.optim import AdamW
from tqdm import tqdm
from .dataset import CaptionDataset
from .model import TransformerBridge, CLIPGPT2CaptionModel
from .utils import save_checkpoint,load_checkpoint
from .config import Config
from huggingface_hub import hf_hub_download

def train(feature_dir, caption_csv, batch_size=16, num_epochs=20, lr=5e-5):
    config = Config()
    device = config.DEVICE

    # Initialize components
    df = pd.read_csv(caption_csv)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = CaptionDataset(df, tokenizer, feature_dir, config.MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model setup
    gpt2_config = GPT2Config.from_pretrained("gpt2")
    gpt2_config.add_cross_attention = True
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", config=gpt2_config)
    bridge = TransformerBridge()
    model = CLIPGPT2CaptionModel(bridge, gpt2)



    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr,weight_decay=0.01)

    #loading weights from huggingface
    checkpoint_path = hf_hub_download(
    repo_id="Kishore0729/image-captioning-model",
    filename="checkpoint_epoch_7.pt",  # or name of file you uploaded
    repo_type="model"
    )
    # Load only the model weights
    model, _, _ = load_checkpoint(
    path=checkpoint_path,
    model=model,
    optimizer=None,  # Pass None to skip optimizer loading
    device="cuda"     # or "cpu"
    )

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
            if loss.dim() != 0:  # Make sure it's a scalar
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Save checkpoint
        checkpoint_path = save_checkpoint(model, optimizer, epoch+1, config.DEFAULT_CHECKPOINT_DIR)
        avg_loss = total_loss / len(dataloader)
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
