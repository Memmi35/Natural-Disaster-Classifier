"""
Visualization utilities for IncidentVision project.

This module provides functions for creating plots, charts, and visualizations
for data exploration, model evaluation, and results presentation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def setup_style():
    """Set up matplotlib and seaborn styling."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def plot_class_distribution(
    labels: List[str], 
    title: str = "Class Distribution",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the distribution of classes in the dataset.
    
    Args:
        labels: List of class labels
        title: Title for the plot
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    bars = ax.bar(unique_labels, counts, alpha=0.8)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Incident Type', fontsize=14)
    ax.set_ylabel('Number of Images', fontsize=14)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: List[int], 
    y_pred: List[int], 
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix with class names.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        title: Title for the plot
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    setup_style()
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training loss values
        val_losses: Validation loss values
        train_accs: Training accuracy values
        val_accs: Validation accuracy values
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    setup_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sample_predictions(
    images: np.ndarray,
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot sample predictions with confidence scores.
    
    Args:
        images: Array of images to display
        true_labels: True class labels
        pred_labels: Predicted class labels
        confidences: Prediction confidence scores
        class_names: Names of the classes
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    setup_style()
    
    n_samples = min(len(images), 8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(n_samples):
        # Denormalize image for display
        img = images[i].transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Create title with prediction info
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        confidence = confidences[i]
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        title = f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}'
        
        axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_interactive_dashboard(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create an interactive dashboard with Plotly.
    
    Args:
        results_df: DataFrame with prediction results
        save_path: Optional path to save the HTML file
        
    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Class Distribution', 'Confidence Distribution', 
                       'Accuracy by Class', 'Prediction Timeline'),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Class distribution
    class_counts = results_df['true_label'].value_counts()
    fig.add_trace(
        go.Bar(x=class_counts.index, y=class_counts.values, name="Class Count"),
        row=1, col=1
    )
    
    # Confidence distribution
    fig.add_trace(
        go.Histogram(x=results_df['confidence'], name="Confidence"),
        row=1, col=2
    )
    
    # Accuracy by class
    accuracy_by_class = results_df.groupby('true_label').apply(
        lambda x: (x['true_label'] == x['pred_label']).mean()
    )
    fig.add_trace(
        go.Bar(x=accuracy_by_class.index, y=accuracy_by_class.values, name="Accuracy"),
        row=2, col=1
    )
    
    # Prediction timeline (if timestamp available)
    if 'timestamp' in results_df.columns:
        fig.add_trace(
            go.Scatter(x=results_df['timestamp'], y=results_df['confidence'], 
                      mode='markers', name="Predictions"),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="IncidentVision Dashboard")
    
    if save_path:
        fig.write_html(save_path)
    
    return fig