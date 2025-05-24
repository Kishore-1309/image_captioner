import torch
from transformers import GPT2Tokenizer, CLIPProcessor, CLIPModel
from PIL import Image
from .model import CLIPGPT2CaptionModel, TransformerBridge
from .config import Config

config = Config()

def generate_caption(
    image_path: str,
    model_path: str = "checkpoints/best_model.pt",
    max_length: int = 30,
    temperature: float = 0.9,
    top_k: int = 50
) -> str:
    """
    Generate caption for an image using trained model
    
    Args:
        image_path: Path to input image
        model_path: Path to saved model checkpoint
        max_length: Maximum caption length
        temperature: Generation temperature (higher = more creative)
        top_k: Top-k sampling parameter
        
    Returns:
        Generated caption string
    """
    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    
    # Load custom model
    bridge = TransformerBridge().to(device)
    model = CLIPGPT2CaptionModel(bridge, gpt2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    
    # Generate caption
    input_ids = tokenizer.encode(tokenizer.bos_token, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids, None, image_features)
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