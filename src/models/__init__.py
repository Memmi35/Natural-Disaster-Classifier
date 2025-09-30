"""
Model architectures and training utilities for IncidentVision.

This module contains the ResNet-18 based classifier and training pipeline
using PyTorch Lightning for natural disaster image classification.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
import torchmetrics
from typing import Dict, Any


class IncidentClassifier(pl.LightningModule):
    """
    ResNet-18 based classifier for incident detection.
    
    Supports 4 classes: on_fire, earthquake, heavy_rainfall, fog
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained ResNet-18
        self.backbone = models.resnet18(pretrained=True)
        
        # Replace the final classifier layer
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, 
            num_classes
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", 
            num_classes=num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", 
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.9
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


def create_model(num_classes: int = 4, **kwargs) -> IncidentClassifier:
    """Factory function to create an IncidentClassifier model."""
    return IncidentClassifier(num_classes=num_classes, **kwargs)