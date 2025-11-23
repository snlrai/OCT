"""
Automated Training Pipeline for OCT Models
Orchestrates training with experiment tracking and model registration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional
import json

from .config import get_config
from .experiment_tracker import ExperimentTracker
from .model_registry import ModelRegistry
from .data_loader import DataLoaderManager


def train_classification_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    tracker: ExperimentTracker,
    model_name: str = 'resnet50'
) -> nn.Module:
    """Train classification model"""
    
    # Initialize model
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 4)  # 4 classes
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: Loss = {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Log metrics
        tracker.log_metrics({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        }, step=epoch)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best model! Validation accuracy: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model


def train_pipeline(
    model_type: str,
    experiment_name: str,
    data_dir: Optional[str] = None,
    use_s3: bool = False,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    use_local_tracking: bool = False
):
    """
    Main training pipeline
    
    Args:
        model_type: 'classification' or 'segmentation'
        experiment_name: Name for the experiment
        data_dir: Local directory with data (or S3 prefix if use_s3=True)
        use_s3: Whether to load data from S3
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_local_tracking: Use local storage instead of AWS
    """
    
    print("=" * 60)
    print("OCT MLOps Training Pipeline")
    print("=" * 60)
    print(f"Model Type: {model_type}")
    print(f"Experiment: {experiment_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("=" * 60)
    
    # Configuration
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        model_type=model_type,
        use_local=use_local_tracking
    )
    
    # Log hyperparameters
    tracker.log_params({
        'model_type': model_type,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'optimizer': 'Adam',
        'device': str(device)
    })
    
    # Data loading
    data_manager = DataLoaderManager(use_s3=use_s3)
    
    if model_type == 'classification':
        class_names = config.classification.class_names
        
        if use_s3:
            data_dir = data_manager.download_data_from_s3('data/classification/', './data/classification')
        elif data_dir is None:
            data_dir = './data/classification'
        
        print(f"\nLoading classification data from: {data_dir}")
        dataloaders = data_manager.get_classification_dataloaders(
            data_dir,
            class_names,
            batch_size=batch_size
        )
        
        # Train
        print("\nStarting training...")
        model = train_classification_model(
            dataloaders['train'],
            dataloaders['val'],
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            tracker=tracker,
            model_name=config.classification.model_name
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in dataloaders['test']:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        # Save model
        output_dir = Path(config.local_models_dir)
        output_dir.mkdir(exist_ok=True)
        model_path = output_dir / 'oct_classifier_mlops.pth'
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Log final metrics
        final_metrics = {'test_accuracy': test_acc}
        
    else:
        raise NotImplementedError("Segmentation training not yet implemented in pipeline")
    
    # Register model
    print("\nRegistering model...")
    registry = ModelRegistry(use_local=use_local_tracking)
    
    model_info = registry.register_model(
        model_name=f'oct-{model_type}',
        model_path=str(model_path),
        model_type=model_type,
        metrics=final_metrics,
        metadata={
            'framework': 'pytorch',
            'architecture': config.classification.model_name if model_type == 'classification' else 'unet',
            'num_epochs': num_epochs,
            'batch_size': batch_size
        },
        experiment_id=tracker.experiment_id
    )
    
    # Log artifact
    tracker.log_artifact('model', model_info['s3_path'], 'model')
    
    # End experiment
    tracker.end_experiment(status='completed', final_metrics=final_metrics)
    
    print("\n" + "=" * 60)
    print("✅ Training pipeline completed successfully!")
    print("=" * 60)
    print(f"Experiment ID: {tracker.experiment_id}")
    print(f"Model Version: {model_info['version']}")
    print(f"Final Metrics: {final_metrics}")
    print("=" * 60)
    
    return model, model_info


def main():
    parser = argparse.ArgumentParser(description='OCT MLOps Training Pipeline')
    parser.add_argument('--model-type', type=str, default='classification',
                        choices=['classification', 'segmentation'],
                        help='Type of model to train')
    parser.add_argument('--experiment-name', type=str, required=True,
                        help='Name for the experiment')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Local directory with training data')
    parser.add_argument('--use-s3', action='store_true',
                        help='Load data from S3')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--local', action='store_true',
                        help='Use local storage instead of AWS')
    
    args = parser.parse_args()
    
    train_pipeline(
        model_type=args.model_type,
        experiment_name=args.experiment_name,
        data_dir=args.data_dir,
        use_s3=args.use_s3,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_local_tracking=args.local
    )


if __name__ == '__main__':
    main()

