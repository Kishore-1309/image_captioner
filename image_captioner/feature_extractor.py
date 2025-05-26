import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Global lazy loading
_model = None
_processor = None
_device = None

def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    global _model, _processor, _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _model is None or _processor is None:
        _model = CLIPModel.from_pretrained(model_name).to(_device)
        _processor = CLIPProcessor.from_pretrained(model_name)
    return _model, _processor, _device

def extract_features_from_image(image_path, save_path):
    model, processor, device = load_clip_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs).squeeze(0).cpu()
    torch.save(features, save_path)
    print(f"Saved features to {save_path}")
    return features

def extract_features_from_folder(image_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for img_name in tqdm(image_files, desc="Extracting features"):
        image_path = os.path.join(image_folder, img_name)
        save_path = os.path.join(save_folder, f"{os.path.splitext(img_name)[0]}.pt")
        extract_features_from_image(image_path, save_path)
    print(f"Saved {len(image_files)} features to {save_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Image Feature Extractor CLI Tool")
    parser.add_argument("--image_path", type=str, help="Path to a single image file")
    parser.add_argument("--save_path", type=str, help="Where to save the .pt feature file for a single image")
    parser.add_argument("--image_folder", type=str, help="Path to a folder of images")
    parser.add_argument("--save_folder", type=str, help="Where to save all extracted features")

    args = parser.parse_args()

    if args.image_path and args.save_path:
        extract_features_from_image(args.image_path, args.save_path)
    elif args.image_folder and args.save_folder:
        extract_features_from_folder(args.image_folder, args.save_folder)
    else:
        print("Please specify either (--image_path and --save_path) or (--image_folder and --save_folder)")
