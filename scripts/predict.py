#!/usr/bin/env python3
"""
Prediction script for IncidentVision model.

This script loads a trained model and makes predictions on new images.
"""

import argparse
import json
import torch
from PIL import Image
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import IncidentClassifier
from features import get_transforms


def load_model(model_path: str, label_mapping_path: str):
    """Load trained model and label mapping."""
    # Load model
    model = IncidentClassifier.load_from_checkpoint(model_path)
    model.eval()
    
    # Load label mapping
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    
    idx_to_label = {v: k for k, v in label_mapping.items()}
    
    return model, idx_to_label


def predict_image(model, image_path: str, transform, idx_to_label: dict):
    """Make prediction on a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()
    
    # Get predicted class name
    predicted_class = idx_to_label[predicted_idx]
    
    # Get all class probabilities
    all_probs = {
        idx_to_label[i]: prob.item() 
        for i, prob in enumerate(probabilities[0])
    }
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': all_probs
    }


def main():
    parser = argparse.ArgumentParser(description='Make predictions with IncidentVision model')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to image file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_model.ckpt',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--label_mapping',
        type=str,
        default='configs/label_mapping.json',
        help='Path to label mapping file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save prediction results (JSON format)'
    )
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.label_mapping):
        print(f"Error: Label mapping file not found: {args.label_mapping}")
        return
    
    print("Loading model...")
    # Load model and label mapping
    model, idx_to_label = load_model(args.model, args.label_mapping)
    
    # Setup transforms
    transform = get_transforms('test')
    
    print(f"Making prediction on: {args.image}")
    # Make prediction
    result = predict_image(model, args.image, transform, idx_to_label)
    
    # Print results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Image: {args.image}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nAll Class Probabilities:")
    
    for class_name, prob in sorted(result['all_probabilities'].items(), 
                                  key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {prob:.2%}")
    
    # Save results if output path specified
    if args.output:
        result['image_path'] = args.image
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("="*50)


if __name__ == '__main__':
    main()