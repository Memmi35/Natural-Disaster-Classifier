#!/usr/bin/env python3
"""
Training script for IncidentVision model.

This script handles the complete training pipeline including data loading,
model initialization, training, and evaluation.
"""

import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import IncidentClassifier
from features import create_dataloaders, save_label_mapping


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_callbacks(config: dict) -> list:
    """Setup PyTorch Lightning callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['paths']['checkpoint_dir'],
        filename='best_model',
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta'],
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    return callbacks


def setup_loggers(config: dict) -> list:
    """Setup PyTorch Lightning loggers."""
    loggers = []
    
    # CSV logger
    csv_logger = CSVLogger(
        save_dir=config['paths']['log_dir'],
        name='training_logs'
    )
    loggers.append(csv_logger)
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config['paths']['log_dir'],
        name='tensorboard_logs'
    )
    loggers.append(tb_logger)
    
    return loggers


def main():
    parser = argparse.ArgumentParser(description='Train IncidentVision model')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/resnet18_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=None,
        help='Number of GPUs to use (None for auto-detection)'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories if they don't exist
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    
    print("Loading data...")
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=config['data']['train_csv'],
        val_csv=config['data']['val_csv'],
        test_csv=config['data']['test_csv'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # Save label mapping
    label_mapping = {label: idx for idx, label in enumerate(config['classes'])}
    save_label_mapping(label_mapping, 'configs/label_mapping.json')
    
    print("Initializing model...")
    # Initialize model
    model = IncidentClassifier(
        num_classes=config['model']['num_classes'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Setup callbacks and loggers
    callbacks = setup_callbacks(config)
    loggers = setup_loggers(config)
    
    print("Setting up trainer...")
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=callbacks,
        logger=loggers,
        accelerator='auto',
        devices=args.gpus,
        precision=16,  # Mixed precision training
        gradient_clip_val=1.0,
        deterministic=True
    )
    
    print("Starting training...")
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    print("Evaluating on test set...")
    # Test the model
    test_results = trainer.test(model, test_loader)
    
    print("Training completed!")
    print(f"Best model saved to: {config['paths']['checkpoint_dir']}/best_model.ckpt")
    print(f"Test results: {test_results}")
    
    # Save final model
    final_model_path = os.path.join(config['paths']['model_dir'], 'final_model.ckpt')
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == '__main__':
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    main()