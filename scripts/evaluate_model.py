#!/usr/bin/env python3
"""
Evaluation script for IncidentVision model.

This script evaluates a trained model on the test set and generates
comprehensive performance metrics and visualizations.
"""

import argparse
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import IncidentClassifier
from features import create_dataloaders, load_label_mapping
from visualization import plot_confusion_matrix, plot_class_distribution


def evaluate_model(model, test_loader, idx_to_label: dict):
    """Evaluate model on test set."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images, labels = batch
            
            # Make predictions
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    return all_predictions, all_labels, all_confidences


def generate_metrics(y_true, y_pred, class_names):
    """Generate comprehensive evaluation metrics."""
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate accuracy
    accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }


def save_predictions(predictions, labels, confidences, idx_to_label, output_path):
    """Save predictions to CSV file."""
    results_df = pd.DataFrame({
        'true_label_idx': labels,
        'pred_label_idx': predictions,
        'true_label': [idx_to_label[idx] for idx in labels],
        'pred_label': [idx_to_label[idx] for idx in predictions],
        'confidence': confidences,
        'correct': np.array(labels) == np.array(predictions)
    })
    
    results_df.to_csv(output_path, index=False)
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Evaluate IncidentVision model')
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_model.ckpt',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--test_csv',
        type=str,
        default='data/processed/test_f.csv',
        help='Path to test CSV file'
    )
    parser.add_argument(
        '--label_mapping',
        type=str,
        default='configs/label_mapping.json',
        help='Path to label mapping file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading model and data...")
    
    # Load model
    model = IncidentClassifier.load_from_checkpoint(args.model)
    model.eval()
    
    # Load label mapping
    label_mapping = load_label_mapping(args.label_mapping)
    idx_to_label = {v: k for k, v in label_mapping.items()}
    class_names = list(label_mapping.keys())
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        train_csv='data/processed/train_f.csv',  # Dummy, not used
        val_csv='data/processed/val_f.csv',      # Dummy, not used
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    print("Evaluating model...")
    
    # Evaluate model
    predictions, labels, confidences = evaluate_model(model, test_loader, idx_to_label)
    
    print("Generating metrics...")
    
    # Generate metrics
    metrics = generate_metrics(labels, predictions, class_names)
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    results_df = save_predictions(predictions, labels, confidences, 
                                idx_to_label, predictions_path)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Generating visualizations...")
    
    # Generate confusion matrix plot
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        labels, predictions, class_names,
        title="Test Set Confusion Matrix",
        save_path=cm_path
    )
    
    # Generate class distribution plot
    dist_path = os.path.join(args.output_dir, 'class_distribution.png')
    plot_class_distribution(
        [idx_to_label[idx] for idx in labels],
        title="Test Set Class Distribution",
        save_path=dist_path
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-Score: {metrics['classification_report']['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {metrics['classification_report']['weighted avg']['f1-score']:.4f}")
    
    print("\nPer-Class Results:")
    for class_name in class_names:
        class_metrics = metrics['classification_report'][class_name]
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1-Score:  {class_metrics['f1-score']:.4f}")
    
    print(f"\nDetailed results saved to: {args.output_dir}")
    print(f"  - Predictions: {predictions_path}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Confusion Matrix: {cm_path}")
    print(f"  - Class Distribution: {dist_path}")
    print("="*60)


if __name__ == '__main__':
    main()