import os
import torch
import tempfile
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


def load_clip_model(model_name="openai/clip-vit-base-patch32", device=None):
    """
    Load the CLIP model and processor.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def get_temp_save_path(filename="temp.pt"):
    """
    Generate a platform-independent temporary file path.
    """
    return os.path.join(tempfile.gettempdir(), filename)


def extract_features_from_image(image_path, save_path=None, model=None, processor=None, device=None):
    """
    Extract CLIP features from a single image, save as .pt file (if save_path given), and return the feature tensor.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        features = model.get_image_features(**inputs).squeeze(0).cpu()  # Shape: [512]

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(features, save_path)
        # print(f"Saved features to {save_path}")
    return features


def extract_features_from_folder(image_folder, save_folder, model, processor, device):
    """
    Extract CLIP features for all images in a folder and save as .pt files.
    """
    os.makedirs(save_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in tqdm(image_files, desc="Extracting features"):
        image_path = os.path.join(image_folder, img_name)
        save_path = os.path.join(save_folder, f"{os.path.splitext(img_name)[0]}.pt")
        extract_features_from_image(image_path, save_path, model, processor, device)

    print(f"Saved {len(image_files)} features to {save_folder}")


if __name__ == "__main__":
    # Load model and processor
    model, processor, device = load_clip_model()

    # Example: Extract features from a folder
    extract_features_from_folder("images", "clip_features", model, processor, device)

    # Example: Extract features from a single image and save to a temp file (cross-platform)
    temp_path = get_temp_save_path("example_temp.pt")
    features = extract_features_from_image("images/example.jpg", temp_path, model, processor, device)
    print("Feature vector shape:", features.shape)
    print(f"Saved temporary features to {temp_path}")
