import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from PIL import Image
from .model import CLIPGPT2CaptionModel, TransformerBridge
from .config import Config
from .utils import load_checkpoint
from .feature_extractor import load_clip_model, extract_features_from_image
from huggingface_hub import hf_hub_download

config = Config()
device = config.DEVICE

def generate_caption(
    image_path: str,
    repo_id: str = "Kishore0729/image-captioning-model",
    filename: str = "checkpoint_epoch_10.pt",
    max_length: int = 30,
    temperature: float = 0.7,
    top_k: int = 50
) -> str:
    """
    Generate caption for an image using trained model downloaded from Hugging Face Hub.

    Args:
        image_path: Path to input image
        repo_id: Hugging Face repo ID where model is stored
        filename: Name of the checkpoint file in the repo
        max_length: Maximum caption length
        temperature: Generation temperature (higher = more creative)
        top_k: Top-k sampling parameter

    Returns:
        Generated caption string
    """

    # Load CLIP model and processor using modular function
    clip_model, processor, device = load_clip_model(device=device)

    # Extract image features using modular function
    feature_tensor = extract_features_from_image(
        image_path=image_path,
        save_path="temp.pt",  # temp file, or you can skip saving
        model=clip_model,
        processor=processor,
        device=device
    ).unsqueeze(0).to(device)  # Shape: [1, 512]

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>"})

    # GPT-2 setup with cross-attention
    gpt2_config = GPT2Config.from_pretrained("gpt2")
    gpt2_config.add_cross_attention = True
    gpt2 = GPT2LMHeadModel(gpt2_config)  # Initialize from config only

    # Build the complete model
    bridge = TransformerBridge()
    model = CLIPGPT2CaptionModel(bridge, gpt2)
    model.gpt2.resize_token_embeddings(len(tokenizer))

    # Load trained checkpoint
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model"
    )
    model, _, _ = load_checkpoint(
        path=model_path,
        model=model,
        optimizer=None,
        device=device
    )
    model.eval()
    model.to(device)

    # Start caption generation
    input_ids = tokenizer.encode(tokenizer.bos_token, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids, None, feature_tensor)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            next_token = torch.multinomial(
                torch.softmax(next_token_logits, dim=-1),
                num_samples=1
            )
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Example usage
    caption = generate_caption("test_image.jpg")
    print("Generated Caption:", caption)
