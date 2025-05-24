# image_captioner/preprocess.py
import os
import re
import json
import pandas as pd
from xml.etree import ElementTree as ET
from typing import Dict, Optional

def clean_caption(text: str) -> str:
    """Clean and normalize caption text."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)         # Replace multiple spaces
    return text

def preprocess_dataset(dataset_name: str, dataset_path: str) -> pd.DataFrame:
    """
    Process raw dataset files into standardized format.
    
    Args:
        dataset_name: One of ['flickr8k', 'flickr30k', 'coco', 'rsicd']
        dataset_path: Path to root directory of the dataset
        
    Returns:
        DataFrame with columns ['image', 'caption']
    """
    data = []
    
    try:
        if dataset_name == 'flickr8k':
            caption_file = os.path.join(dataset_path, 'Flickr8k.token.txt')
            with open(caption_file, 'r') as f:
                for line in f:
                    if '\t' in line:
                        img_name, caption = line.strip().split('\t')
                        img_name = img_name.split('#')[0]
                        data.append({
                            'image': img_name,
                            'caption': clean_caption(caption)
                        })

        elif dataset_name == 'flickr30k':
            caption_file = os.path.join(dataset_path, 'results.csv')
            df = pd.read_csv(caption_file, sep='|', header=0)
            df = df.rename(columns={'image_name': 'image', 'comment': 'caption'})
            df['caption'] = df['caption'].apply(clean_caption)
            data = df[['image', 'caption']].to_dict('records')

        elif dataset_name == 'coco':
            ann_file = os.path.join(dataset_path, 'captions_train2017.json')
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
            
            image_id_map = {img['id']: img['file_name'] 
                          for img in annotations['images']}
            
            for ann in annotations['annotations']:
                data.append({
                    'image': image_id_map[ann['image_id']],
                    'caption': clean_caption(ann['caption'])
                })

        elif dataset_name == 'rsicd':
            ann_dir = os.path.join(dataset_path, 'Annotations')
            for xml_file in os.listdir(ann_dir):
                if xml_file.endswith('.xml'):
                    tree = ET.parse(os.path.join(ann_dir, xml_file))
                    root = tree.getroot()
                    
                    img_name = root.find('filename').text
                    captions = [obj.find('name').text 
                               for obj in root.findall('object')]
                    
                    # Use first caption per image
                    if captions:
                        data.append({
                            'image': img_name,
                            'caption': clean_caption(captions[0])
                        })

    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
    
    return pd.DataFrame(data)

def process_all_datasets(dataset_paths: Dict[str, str], 
                        output_path: str = "processed_captions.csv") -> pd.DataFrame:
    """
    Process multiple datasets and merge into a single DataFrame.
    
    Args:
        dataset_paths: Dictionary mapping dataset names to paths
            Example: {'coco': '/path/to/coco', 'flickr8k': '/path/to/flickr8k'}
        output_path: Path to save combined CSV file
    
    Returns:
        Combined DataFrame with deduplicated captions
    """
    all_data = []
    
    for name, path in dataset_paths.items():
        print(f"Processing {name} dataset...")
        df = preprocess_dataset(name, path)
        if not df.empty:
            all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['image'])
    
    # Save to CSV
    combined_df.to_csv(output_path, index=False)
    print(f"Saved processed captions to {output_path}")
    
    return combined_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess image caption datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (coco, flickr8k, flickr30k, rsicd)')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to raw dataset directory')
    parser.add_argument('--output_path', type=str, default='processed_captions.csv',
                       help='Output path for processed CSV file')
    
    args = parser.parse_args()
    
    # Process single dataset
    df = preprocess_dataset(args.dataset, args.input_path)
    df.to_csv(args.output_path, index=False)
    print(f"Processed {len(df)} captions for {args.dataset}")