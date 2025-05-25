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
        # Verify file exists and is readable
        if not os.path.exists(caption_file_path):
            raise FileNotFoundError(f"File not found: {caption_file_path}")
        
        print(f"Reading captions from: {caption_file_path}")
        
        with open(caption_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                if '\t' in line:
                    img_name, caption = line.split('\t')
                    img_name = img_name.split('#')[0]  # Remove suffix (e.g., '#0')
                    data.append({
                        'image': img_name,
                        'caption': clean_caption(caption)
                    })
                else:
                    print(f"Warning: Line {line_num} has no tab separator: '{line}'")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            print("⚠️ No valid captions found. Check input file format.")
        else:
            # Save to CSV
            df.to_csv(output_csv_path, index=False)
            print(f"✅ Successfully processed {len(df)} captions. Saved to: {output_csv_path}")
        
        return df
    
    except Exception as e:
        print(f"❌ Error processing Flickr8K dataset: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Flickr8K captions into a cleaned CSV')
    parser.add_argument('--caption_file', type=str, required=True,
                       help='Path to Flickr8K caption.txt file (e.g., "/kaggle/input/flickr8k/caption.txt")')
    parser.add_argument('--output_csv', type=str, default='flickr8k_processed.csv',
                       help='Output path for processed CSV (default: flickr8k_processed.csv)')
    
    args = parser.parse_args()
    
    # Process the dataset
    df = process_flickr8k(args.caption_file, args.output_csv)
    
    # Debug: Print first 5 rows if data exists
    if not df.empty:
        print("\nSample processed captions:")
        print(df.head())