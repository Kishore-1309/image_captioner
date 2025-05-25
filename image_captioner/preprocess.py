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
    Process Flickr8K dataset from CSV caption file to cleaned CSV.
    
    Args:
        caption_file_path: Path to Flickr8K's captions.csv file
        output_csv_path: Path to save the processed CSV
    """
    try:
        # Verify file exists
        if not os.path.exists(caption_file_path):
            raise FileNotFoundError(f"File not found: {caption_file_path}")
        
        print(f"Reading captions from: {caption_file_path}")
        
        # Read as CSV (handles both header and no-header cases)
        df = pd.read_csv(caption_file_path)
        
        # Standardize column names (case-insensitive)
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Check required columns
        if not all(col in df.columns for col in ['image', 'caption']):
            raise ValueError("CSV must contain 'image' and 'caption' columns")
        
        # Clean and process
        df['caption'] = df['caption'].apply(clean_caption)
        df['image'] = df['image'].str.split('#').str[0]  # Remove suffixes like #0
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['image', 'caption'])
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        print(f"✅ Processed {len(df)} captions. Saved to: {output_csv_path}")
        return df
        
    except Exception as e:
        print(f"❌ Error processing dataset: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Flickr8K captions CSV')
    parser.add_argument('--caption_file', required=True, help='Path to captions.csv')
    parser.add_argument('--output_csv', default='flickr8k_processed.csv', help='Output CSV path')
    
    args = parser.parse_args()
    df = process_flickr8k(args.caption_file, args.output_csv)
    
    if not df.empty:
        print("\nSample output:")
        print(df.head(3))