import os
import torch
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


def extract_features_from_image(image_path, model, processor, device):
    """
    Extract CLIP features from a single image, save as .pt file, and return the feature tensor.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        features = model.get_image_features(**inputs).squeeze(0).cpu()  # Shape: [512]

    #torch.save(features, save_path)
    #print(f"Saved features to {save_path}")
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
        extract_features_from_image(image_path, model, processor, device)

    print(f"Saved {len(image_files)} features to {save_folder}")


if __name__ == "__main__":
    # Example usage:
    model, processor, device = load_clip_model()

    # For folder
    extract_features_from_folder("images", "clip_features", model, processor, device)

    # For single image (also returns the feature tensor)
    features = extract_features_from_image("images/example.jpg", model, processor, device)
    # print("Feature vector shape:", features.shape)
