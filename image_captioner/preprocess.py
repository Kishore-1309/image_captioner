import os
import re
import pandas as pd

def clean_caption(text: str) -> str:
    """Clean and normalize caption text."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)         # Replace multiple spaces
    return text

def process_flickr8k(caption_file_path: str, output_csv_path: str = "flickr8k_processed.csv") -> pd.DataFrame:
    """
    Process Flickr8K dataset from raw caption file to cleaned CSV.
    
    Args:
        caption_file_path: Path to Flickr8K's caption.txt file
        output_csv_path: Path to save the processed CSV (default: 'flickr8k_processed.csv')
        
    Returns:
        DataFrame with columns ['image', 'caption']
    """
    data = []
    
    try:
        with open(caption_file_path, 'r') as f:
            for line in f:
                if '\t' in line:
                    img_name, caption = line.strip().split('\t')
                    img_name = img_name.split('#')[0]  # Remove suffix (e.g., '#0')
                    data.append({
                        'image': img_name,
                        'caption': clean_caption(caption)
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully processed {len(df)} captions. Saved to: {output_csv_path}")
        return df
    
    except Exception as e:
        print(f"Error processing Flickr8K dataset: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Flickr8K captions into a cleaned CSV')
    parser.add_argument('--caption_file', type=str, required=True,
                       help='Path to Flickr8K caption.txt file')
    parser.add_argument('--output_csv', type=str, default='flickr8k_processed.csv',
                       help='Output path for processed CSV (default: flickr8k_processed.csv)')
    
    args = parser.parse_args()
    
    # Process the dataset
    df = process_flickr8k(args.caption_file, args.output_csv)