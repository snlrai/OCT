"""
Data Loader for OCT Images
Supports loading from local storage or S3
"""

import os
import boto3
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import json
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import get_config


class OCTClassificationDataset(Dataset):
    """Dataset for OCT classification"""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
        use_augmentation: bool = True
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_augmentation = use_augmentation
        
        # Default augmentation
        if use_augmentation and transform is None:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label


class OCTSegmentationDataset(Dataset):
    """Dataset for OCT segmentation"""
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform=None,
        image_size: Tuple[int, int] = (256, 256)
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_size = image_size
        
        # Default transform
        if transform is None:
            self.transform = A.Compose([
                A.Resize(*image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load image (grayscale)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return image, mask


class DataLoaderManager:
    """Manages data loading from local or S3"""
    
    def __init__(self, use_s3: bool = False):
        self.config = get_config()
        self.use_s3 = use_s3
        
        if use_s3:
            self.s3_client = boto3.client('s3', region_name=self.config.aws.region)
    
    def download_data_from_s3(self, s3_prefix: str, local_dir: str) -> str:
        """
        Download data from S3 to local directory
        
        Args:
            s3_prefix: S3 prefix (e.g., 'data/classification/')
            local_dir: Local directory to save data
        
        Returns:
            Path to local directory
        """
        bucket = self.config.aws.s3_data_bucket
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading data from s3://{bucket}/{s3_prefix}...")
        
        # List objects
        response = self.s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=s3_prefix
        )
        
        if 'Contents' not in response:
            print(f"No objects found at s3://{bucket}/{s3_prefix}")
            return str(local_path)
        
        # Download each file
        for obj in response['Contents']:
            s3_key = obj['Key']
            relative_path = s3_key[len(s3_prefix):].lstrip('/')
            local_file = local_path / relative_path
            
            # Create parent directories
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Download
            self.s3_client.download_file(bucket, s3_key, str(local_file))
        
        print(f"✓ Downloaded {len(response['Contents'])} files")
        return str(local_path)
    
    def prepare_classification_data(
        self,
        data_dir: str,
        class_names: List[str],
        train_split: float = 0.8,
        val_split: float = 0.1
    ) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Prepare classification data
        
        Args:
            data_dir: Directory containing class subdirectories
            class_names: List of class names
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
        
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        data_path = Path(data_dir)
        
        all_images = []
        all_labels = []
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = data_path / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            # Get all image files
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
            
            all_images.extend([str(img) for img in images])
            all_labels.extend([class_idx] * len(images))
        
        # Shuffle
        indices = np.random.permutation(len(all_images))
        all_images = [all_images[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
        
        # Split
        n = len(all_images)
        train_end = int(n * train_split)
        val_end = train_end + int(n * val_split)
        
        splits = {
            'train': (all_images[:train_end], all_labels[:train_end]),
            'val': (all_images[train_end:val_end], all_labels[train_end:val_end]),
            'test': (all_images[val_end:], all_labels[val_end:])
        }
        
        print(f"Data split: train={len(splits['train'][0])}, val={len(splits['val'][0])}, test={len(splits['test'][0])}")
        
        return splits
    
    def prepare_segmentation_data(
        self,
        images_dir: str,
        masks_dir: str,
        train_split: float = 0.8,
        val_split: float = 0.1
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Prepare segmentation data
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
        
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        images_path = Path(images_dir)
        masks_path = Path(masks_dir)
        
        # Get all image files
        images = sorted(list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')))
        
        # Match with masks
        image_mask_pairs = []
        for img_path in images:
            mask_path = masks_path / img_path.name
            if mask_path.exists():
                image_mask_pairs.append((str(img_path), str(mask_path)))
        
        # Shuffle
        np.random.shuffle(image_mask_pairs)
        
        # Split
        n = len(image_mask_pairs)
        train_end = int(n * train_split)
        val_end = train_end + int(n * val_split)
        
        splits = {
            'train': ([p[0] for p in image_mask_pairs[:train_end]], [p[1] for p in image_mask_pairs[:train_end]]),
            'val': ([p[0] for p in image_mask_pairs[train_end:val_end]], [p[1] for p in image_mask_pairs[train_end:val_end]]),
            'test': ([p[0] for p in image_mask_pairs[val_end:]], [p[1] for p in image_mask_pairs[val_end:]])
        }
        
        print(f"Data split: train={len(splits['train'][0])}, val={len(splits['val'][0])}, test={len(splits['test'][0])}")
        
        return splits
    
    def get_classification_dataloaders(
        self,
        data_dir: str,
        class_names: List[str],
        batch_size: int = 32,
        num_workers: int = 2
    ) -> Dict[str, DataLoader]:
        """
        Get DataLoaders for classification
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        """
        # Prepare data splits
        splits = self.prepare_classification_data(data_dir, class_names)
        
        # Create datasets
        datasets = {
            'train': OCTClassificationDataset(*splits['train'], use_augmentation=True),
            'val': OCTClassificationDataset(*splits['val'], use_augmentation=False),
            'test': OCTClassificationDataset(*splits['test'], use_augmentation=False)
        }
        
        # Create dataloaders
        dataloaders = {
            split: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True
            )
            for split, dataset in datasets.items()
        }
        
        return dataloaders
    
    def get_segmentation_dataloaders(
        self,
        images_dir: str,
        masks_dir: str,
        batch_size: int = 8,
        num_workers: int = 2
    ) -> Dict[str, DataLoader]:
        """
        Get DataLoaders for segmentation
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        """
        # Prepare data splits
        splits = self.prepare_segmentation_data(images_dir, masks_dir)
        
        # Create datasets
        datasets = {
            split: OCTSegmentationDataset(images, masks)
            for split, (images, masks) in splits.items()
        }
        
        # Create dataloaders
        dataloaders = {
            split: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True
            )
            for split, dataset in datasets.items()
        }
        
        return dataloaders

