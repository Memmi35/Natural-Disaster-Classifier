#!/usr/bin/env python3
"""
Data preprocessing script for IncidentVision.

This script handles data download, validation, and preprocessing tasks.
"""

import argparse
import json
import pandas as pd
import requests
from PIL import Image
import os
from tqdm import tqdm
import shutil
from urllib.parse import urlparse
import hashlib


def download_image(url: str, filepath: str, timeout: int = 30) -> bool:
    """Download image from URL and save to filepath."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        
        # Verify image can be opened
        Image.open(filepath).verify()
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def process_incident_json(json_file: str, output_dir: str, categories: list) -> pd.DataFrame:
    """Process Incident1M JSON file and download images."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Filter data for specified categories
    filtered_data = [
        item for item in data 
        if item.get('category') in categories
    ]
    
    print(f"Found {len(filtered_data)} images in specified categories")
    
    # Create output directories
    for category in categories:
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)
    
    successful_downloads = []
    
    for item in tqdm(filtered_data, desc="Downloading images"):
        category = item['category']
        filename = item['filename']
        url = item['url']
        
        # Generate filepath
        file_ext = os.path.splitext(filename)[1] or '.jpg'
        safe_filename = hashlib.md5(url.encode()).hexdigest() + file_ext
        filepath = os.path.join(output_dir, category, safe_filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            successful_downloads.append({
                'filepath': filepath,
                'category': category,
                'original_filename': filename,
                'url': url
            })
            continue
        
        # Download image
        if download_image(url, filepath):
            successful_downloads.append({
                'filepath': filepath,
                'category': category,
                'original_filename': filename,
                'url': url
            })
    
    print(f"Successfully downloaded {len(successful_downloads)} images")
    
    return pd.DataFrame(successful_downloads)


def create_train_val_test_split(df: pd.DataFrame, output_dir: str, 
                               train_ratio: float = 0.7, 
                               val_ratio: float = 0.2):
    """Create stratified train/validation/test split."""
    from sklearn.model_selection import train_test_split
    
    # Rename column to match expected format
    df_split = df.copy()
    df_split['label'] = df_split['category']
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df_split, 
        test_size=1 - train_ratio - val_ratio,
        stratify=df_split['label'],
        random_state=42
    )
    
    # Second split: separate train and validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio / (train_ratio + val_ratio),
        stratify=train_val_df['label'],
        random_state=42
    )
    
    # Save CSV files
    os.makedirs(output_dir, exist_ok=True)
    
    train_df[['filepath', 'label']].to_csv(
        os.path.join(output_dir, 'train_f.csv'), index=False
    )
    val_df[['filepath', 'label']].to_csv(
        os.path.join(output_dir, 'val_f.csv'), index=False
    )
    test_df[['filepath', 'label']].to_csv(
        os.path.join(output_dir, 'test_f.csv'), index=False
    )
    
    print("Data split summary:")
    print(f"Train: {len(train_df)} images")
    print(f"Validation: {len(val_df)} images")
    print(f"Test: {len(test_df)} images")
    
    return train_df, val_df, test_df


def validate_images(image_dir: str) -> list:
    """Validate all images in directory and remove corrupted ones."""
    corrupted_files = []
    
    for root, dirs, files in os.walk(image_dir):
        for file in tqdm(files, desc=f"Validating {root}"):
            filepath = os.path.join(root, file)
            
            try:
                with Image.open(filepath) as img:
                    img.verify()
            except Exception as e:
                print(f"Corrupted image: {filepath} - {e}")
                corrupted_files.append(filepath)
                os.remove(filepath)
    
    print(f"Removed {len(corrupted_files)} corrupted images")
    return corrupted_files


def main():
    parser = argparse.ArgumentParser(description='Preprocess data for IncidentVision')
    parser.add_argument(
        '--json_file',
        type=str,
        required=True,
        help='Path to Incident1M JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/raw',
        help='Output directory for downloaded images'
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        default='data/processed',
        help='Output directory for processed CSV files'
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        default=['on_fire', 'earthquake', 'heavy_rainfall', 'fog'],
        help='Categories to include'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate downloaded images'
    )
    args = parser.parse_args()
    
    print("Starting data preprocessing...")
    
    # Process JSON and download images
    df = process_incident_json(args.json_file, args.output_dir, args.categories)
    
    # Validate images if requested
    if args.validate:
        print("Validating images...")
        corrupted = validate_images(args.output_dir)
        
        # Remove corrupted files from DataFrame
        df = df[~df['filepath'].isin(corrupted)]
    
    # Create train/val/test split
    print("Creating train/validation/test split...")
    train_df, val_df, test_df = create_train_val_test_split(df, args.processed_dir)
    
    # Print category distribution
    print("\nCategory distribution:")
    for category in args.categories:
        count = len(df[df['category'] == category])
        print(f"  {category}: {count} images")
    
    print(f"\nData preprocessing completed!")
    print(f"Raw images saved to: {args.output_dir}")
    print(f"Processed CSV files saved to: {args.processed_dir}")


if __name__ == '__main__':
    main()