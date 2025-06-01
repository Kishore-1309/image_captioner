import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from huggingface_hub import hf_hub_download

from .model import CLIPGPT2CaptionModel, TransformerBridge
from .config import Config
from .utils import load_checkpoint
from .feature_extractor import load_clip_model, extract_features_from_image

config = Config()
device = config.DEVICE

def generate_caption(
    image_path: str,
    repo_id: str = "Kishore0729/image-captioning-model",
    filename: str = "checkpoint_epoch_6.pt",
    max_length: int = 40,
    temperature: float = 0.7,
    top_k: int = 50
) -> str:
    """
    Generate a caption for a given image using a trained CLIP-GPT2 model.

    Args:
        image_path: Path to input image
        repo_id: Hugging Face repo ID where model is stored
        filename: Checkpoint filename
        max_length: Max caption length
        temperature: Sampling temperature
        top_k: Top-k sampling

    Returns:
        Generated caption string
    """

    # Load CLIP model and processor
    clip_model, processor, _ = load_clip_model()

    # Extract image features
    image_tensor = extract_features_from_image(
        image_path=image_path,
        save_path="/tmp/temp.pt",  # Temporary save path
        model=clip_model,
        processor=processor,
        device=device
    ).unsqueeze(0).to(device)  # Shape: [1, 512]

    # Load tokenizer and GPT-2 model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    
    gpt2_config = GPT2Config.from_pretrained("gpt2")
    gpt2_config.add_cross_attention = True
    gpt2 = GPT2LMHeadModel(gpt2_config)

    # Build the CLIP-GPT2 model
    bridge = TransformerBridge()
    model = CLIPGPT2CaptionModel(bridge, gpt2)

    # Load the trained weights from Hugging Face
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    model, _, _ = load_checkpoint(model_path, model, optimizer=None, device=device)

    # Resize token embeddings AFTER loading weights
    model.gpt2.resize_token_embeddings(len(tokenizer))

    model.eval()
    model.to(device)

    # Generate caption
    input_ids = tokenizer.encode(tokenizer.bos_token, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids, attention_mask, image_tensor)
            logits = outputs.logits[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.ones_like(input_ids).to(device)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    test_image = "/kaggle/input/flickr8k/Images/1000268201_693b08cb0e.jpg"
    generated_caption = generate_caption(test_image)
    print("Generated Caption:", generated_caption)
     
