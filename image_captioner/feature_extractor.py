import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

def extract_features(image_folder, save_folder, model_name="openai/clip-vit-base-patch32"):
    os.makedirs(save_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_name in tqdm(image_files, desc="Extracting features"):
        image_path = os.path.join(image_folder, img_name)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            features = model.get_image_features(**inputs).cpu()
            
        torch.save(features, os.path.join(save_folder, f"{os.path.splitext(img_name)[0]}.pt"))
    
    print(f"Saved {len(image_files)} features to {save_folder}")