"""
Data preprocessing and feature extraction utilities for IncidentVision.

This module handles dataset creation, image transformations, and data loading
for the incident classification pipeline.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple, Optional
import json


class IncidentDataset(Dataset):
    """
    Custom dataset class for incident images.
    
    Args:
        csv_file (str): Path to CSV file with image paths and labels
        transform (callable, optional): Optional transform to be applied on images
        label_mapping (dict, optional): Mapping from label names to indices
    """
    
    def __init__(
        self, 
        csv_file: str, 
        transform: Optional[transforms.Compose] = None,
        label_mapping: Optional[Dict[str, int]] = None
    ):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        
        # Create label mapping if not provided
        if label_mapping is None:
            unique_labels = self.data_frame['label'].unique()
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_mapping = label_mapping
            
        # Reverse mapping for inference
        self.idx_to_label = {idx: label for label, idx in self.label_mapping.items()}
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.data_frame.iloc[idx]['filepath']
        label_name = self.data_frame.iloc[idx]['label']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert label to index
        label_idx = self.label_mapping[label_name]
        
        return image, label_idx


def get_transforms(mode: str = 'train') -> transforms.Compose:
    """
    Get image transformations for training or inference.
    
    Args:
        mode (str): Either 'train', 'val', or 'test'
        
    Returns:
        transforms.Compose: Composed transformations
    """
    
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_csv (str): Path to training CSV
        val_csv (str): Path to validation CSV  
        test_csv (str): Path to test CSV
        batch_size (int): Batch size for data loading
        num_workers (int): Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Define transforms
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    # Create datasets
    train_dataset = IncidentDataset(train_csv, transform=train_transform)
    val_dataset = IncidentDataset(val_csv, transform=val_transform, 
                                  label_mapping=train_dataset.label_mapping)
    test_dataset = IncidentDataset(test_csv, transform=val_transform,
                                   label_mapping=train_dataset.label_mapping)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def save_label_mapping(label_mapping: Dict[str, int], filepath: str):
    """Save label mapping to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(label_mapping, f, indent=2)


def load_label_mapping(filepath: str) -> Dict[str, int]:
    """Load label mapping from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)