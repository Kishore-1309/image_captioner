import os
import csv
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from huggingface_hub import hf_hub_download

from .model import CLIPGPT2CaptionModel, TransformerBridge
from .config import Config
from .utils import load_checkpoint

config = Config()
device = config.DEVICE

def generate_caption_from_tensor(
    image_tensor: torch.Tensor,
    model: CLIPGPT2CaptionModel,
    tokenizer: GPT2Tokenizer,
    max_length: int = 40,
    temperature: float = 0.7,
    top_k: int = 50
) -> str:
    """
    Generate a caption from a CLIP image tensor using a trained CLIP-GPT2 model.
    """
    model.eval()
    input_ids = tokenizer.encode(tokenizer.bos_token, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids, None, image_tensor)
            logits = outputs.logits[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def generate_captions_for_folder(
    features_folder: str,
    output_csv: str = "captions.csv",
    repo_id: str = "Kishore0729/image-captioning-model",
    filename: str = "model_3.pt"
):
    """
    Generate captions for all image features (.pt) in a folder and save to CSV.
    """
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>"})

    gpt2_config = GPT2Config.from_pretrained("gpt2")
    gpt2_config.add_cross_attention = True
    gpt2 = GPT2LMHeadModel(gpt2_config)

    bridge = TransformerBridge()
    model = CLIPGPT2CaptionModel(bridge, gpt2)

    # Load trained weights
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    model, _, _ = load_checkpoint(model_path, model, optimizer=None, device=device)
    model.gpt2.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Prepare output
    results = []

    # Iterate through all .pt files in the folder
    for fname in os.listdir(features_folder):
        if fname.endswith(".pt"):
            image_id = os.path.splitext(fname)[0]
            tensor_path = os.path.join(features_folder, fname)
            image_tensor = torch.load(tensor_path).unsqueeze(0).to(device)  # Shape: [1, 512]
            caption = generate_caption_from_tensor(image_tensor, model, tokenizer)
            results.append({"image_id": image_id, "caption": caption})
            print(f"Caption generated for {image_id}: {caption}")

    # Write to CSV
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "caption"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCaptions saved to {output_csv}")


if __name__ == "__main__":
    feature_folder_path = "/kaggle/input/flickr8k_clip_features"
    output_csv_path = "/kaggle/working/captions.csv"
    generate_captions_for_folder(features_folder=feature_folder_path, output_csv=output_csv_path)
