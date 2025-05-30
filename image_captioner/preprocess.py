import os
import re
import json
import pandas as pd

def clean_caption(text: str) -> str:
    """Clean and normalize caption text."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)         # Replace multiple spaces
    return text

def process_flickr8k_csv(caption_file_path: str) -> pd.DataFrame:
    """Process Flickr8K CSV caption format."""
    df = pd.read_csv(caption_file_path)

    # Normalize column names
    df.columns = [col.lower().strip() for col in df.columns]
    if not all(col in df.columns for col in ['image', 'caption']):
        raise ValueError("CSV must contain 'image' and 'caption' columns")

    df['caption'] = df['caption']
    df['image'] = df['image'].str.split('#').str[0]  # Remove suffixes like #0
    df = df.drop_duplicates(subset=['image', 'caption'])
    return df

def process_coco_json(json_path: str) -> pd.DataFrame:
    """Process COCO JSON caption format."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    images = data.get("images", [])

    # Map image_id to filename
    image_id_to_filename = {img["id"]: img["file_name"] for img in images}

    rows = []
    for ann in annotations:
        image_id = ann["image_id"]
        caption = ann["caption"]
        filename = image_id_to_filename.get(image_id, f"{image_id}.jpg")
        rows.append((filename, caption))

    df = pd.DataFrame(rows, columns=["image", "caption"])
    df = df.drop_duplicates(subset=['image', 'caption'])
    return df

def process_caption_file(input_path: str, output_csv_path: str = "processed_captions.csv") -> pd.DataFrame:
    """
    General function to process either Flickr8K (CSV) or COCO (JSON) caption files.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".txt":
        df = process_flickr8k_csv(input_path)
    elif ext == ".json":
        df = process_coco_json(input_path)
    else:
        raise ValueError("Unsupported file type. Use CSV for Flickr8K or JSON for COCO.")

    df.to_csv(output_csv_path, index=False)
    print(f"✅ Processed {len(df)} captions. Saved to: {output_csv_path}")
    return df

# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process Flickr8K or COCO caption files.')
    parser.add_argument('--input_path', required=True, help='Path to input caption file (CSV for Flickr8K or JSON for COCO)')
    parser.add_argument('--output_csv', default='processed_captions.csv', help='Path to save the cleaned output CSV')

    args = parser.parse_args()
    df = process_caption_file(args.input_path, args.output_csv)

    if not df.empty:
        print("\nSample output:")
        print(df.head(3))
